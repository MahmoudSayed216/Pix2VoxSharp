import torch.optim.adadelta
import yaml
from Dataset2 import ShapeNetDataset
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from model import full_model
from utils.debugger import DEBUGGER_SINGLETON, DEBUG, LOG, CHECKPOINT
import os
from datetime import datetime
from metrics.loss import VoxelLoss
from metrics.IoU import compute_iou
from utils import network_utils
from writer import Writer
import numpy as np



def compute_validation_metrics(model, loss_fn, loader, THRESHOLDS, ITERATIONS_PER_EPOCH, configs):
    model.eval()
    VAL_LOSS_ACCUMULATOR = 0

    IOU_20 = []
    IOU_25 = []
    IOU_30 = []
    IOU_35 = []
    IOU_40 = []

    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, volumes = batch
            images = images.to(configs["device"])
            volumes = volumes.to(configs["device"])

            outputs = model(images)

            loss = loss_fn(outputs, volumes)
            VAL_LOSS_ACCUMULATOR+=loss.item()
            iou20, iou25, iou30, iou35, iou40 = compute_iou(outputs, volumes, ths=THRESHOLDS)  ## ASSUMING THEY ARE 3
            IOU_20.append(iou20)
            IOU_25.append(iou25)
            IOU_30.append(iou30)
            IOU_35.append(iou35)
            IOU_40.append(iou40)

        mean_val_loss = VAL_LOSS_ACCUMULATOR/ITERATIONS_PER_EPOCH
        mean_IOU_20 = sum(IOU_20)/len(IOU_20)
        mean_IOU_25 = sum(IOU_25)/len(IOU_25)
        mean_IOU_30 = sum(IOU_30)/len(IOU_30)
        mean_IOU_35 = sum(IOU_35)/len(IOU_35)
        mean_IOU_40 = sum(IOU_40)/len(IOU_40)

    return mean_val_loss, [mean_IOU_20, mean_IOU_25, mean_IOU_30, mean_IOU_35, mean_IOU_40]

def gaussian_random(low=1, high=12):
    mu = 6.5
    sigma = 3.5
    while True:
        x = np.random.normal(mu, sigma)  # Generate a Gaussian sample
        if low <= x <= high:  # Accept only if within range
            return int(round(x))  # Convert to integer



def update_dataset_configs(loader):
    random_value = gaussian_random(1, 12)
    loader.dataset.set_n_views_rendering(random_value)

    # loader.dataset.choose_images_indices_for_epoch()
    return random_value



def train(configs):
    writer = Writer(configs["train_path"])
    writer.add_line(str(configs))
    data_path = configs["dataset"]["data"]
    json_file_path = configs["dataset"]["json_mapper"]


    train_transforms = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomApply([
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    ], p=0.8),
    T.RandomGrayscale(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),  # Resize shorter side to 256
        T.CenterCrop(224),  # Crop the center to 224x224
        T.ToTensor(),  # Convert to tensor
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])

    BATCH_SIZE = configs["train"]["batch_size"]
    DEBUG("BATCH SIZE", BATCH_SIZE)

    train_dataset = ShapeNetDataset(data_path, json_file_path, split='train', transforms=train_transforms)
    train_dataset.set_n_views_rendering(1)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, drop_last=True)

    val_dataset = ShapeNetDataset(data_path, json_file_path, split='val', transforms=val_transform)
    val_dataset.set_n_views_rendering(1)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, drop_last=True)




    EPOCHS = configs["train"]["epochs"]
    START_EPOCH = configs["train"]["start_epoch"]
    THRESHOLDS = configs["thresholds"]

    model = full_model.Pix2VoxSharp(configs).to(configs["device"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=trainable_params, lr=configs["optim"]["lr"])
    loss_fn = VoxelLoss(weight=10)
    ITERATIONS_PER_EPOCH_TRAIN = int(len(train_dataset)/BATCH_SIZE)
    ITERATIONS_PER_EPOCH_VAL = int(len(val_dataset)/BATCH_SIZE)
    current_best_iou = 0
    for epoch in range(START_EPOCH, EPOCHS):
        LOG("TRAINING")
        LOG("EPOCH", epoch+1)
        writer.add_line(f"EPOCH: {epoch+1}")
        model.train()
        TRAIN_LOSS_ACCUMUlATOR = 0
        for idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            images, volumes = batch
            images = images.to(configs["device"])
            volumes = volumes.to(configs["device"])

            outputs = model(images)
            loss = loss_fn(outputs, volumes)
            TRAIN_LOSS_ACCUMUlATOR += loss.item()
            loss.backward()
            optimizer.step()
            if idx % 50 == 0:

                LOG("current loss", loss.item())

        average_epoch_loss = TRAIN_LOSS_ACCUMUlATOR/ITERATIONS_PER_EPOCH_TRAIN

        LOG("TESTING")
        valid_loss, valid_IoU = compute_validation_metrics(model, loss_fn, val_loader, THRESHOLDS, ITERATIONS_PER_EPOCH_VAL, configs)
        mean_iou = sum(valid_IoU)/len(valid_IoU)

        LOG("average train loss", average_epoch_loss)
        LOG("average test loss", valid_loss)
        LOG("test IoU @ different THs", valid_IoU)
        LOG("mean test IoU", mean_iou)
        


        if mean_iou > current_best_iou:
            current_best_iou = mean_iou
            CHECKPOINT(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")
            writer.add_line(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")
            weights_path = os.path.join(configs["train_path"], "weights", "best.pth")
            network_utils.save_checkpoints(weights_path, epoch+1,model, optimizer, mean_iou, epoch+1)
            samples_path = os.path.join(configs["train_path"], "samples", f"output{epoch+1}.pth")
            torch.save(outputs, samples_path)

            LOG("tensor saved")
            
        if (epoch+1) % configs["train"]["save_every"] == 0:
            weights_path = os.path.join(configs["train_path"], "weights", "last.pth")
            CHECKPOINT("Saving last Weights...")
            network_utils.save_checkpoints(weights_path, epoch+1,model, optimizer, mean_iou, epoch+1)

            if (epoch+1) % configs["train"]["reduce_lr_epoch"]== 0:
                LOG("REDUCING LR")
                reduce_lr_factor = configs["train"]["reduce_lr_factor"]
                learning_rate*= reduce_lr_factor
                writer.add_line(f"Learning rate has been reduced to {learning_rate} at epoch {epoch+1}")
                writer.add_line(f"")
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= reduce_lr_factor

        if (epoch+1) == configs["train"]["epochs_till_merger"]:
            model.set_merger(True)
            LOG("MERGER ACTIVATED")
            writer.add_line("MERGER ACTIVATED")

        if (epoch+1) >= configs["train"]["epochs_till_merger"]:
            random_val = gaussian_random(1, 12)
            train_loader.dataset.set_n_views_rendering(random_val)
            val_loader.dataset.set_n_views_rendering(random_val)

        
        writer.add_scaler("TRAIN LOSS", epoch+1, average_epoch_loss)
        writer.add_scaler("VALID LOSS", epoch+1, valid_loss)
        writer.add_scaler("VALID IoU@20", epoch+1, valid_IoU[0])
        writer.add_scaler("VALID IoU@25", epoch+1, valid_IoU[1])
        writer.add_scaler("VALID IoU@30", epoch+1, valid_IoU[2])
        writer.add_scaler("VALID IoU@35", epoch+1, valid_IoU[3])
        writer.add_scaler("VALID IoU@40", epoch+1, valid_IoU[4])
        writer.add_scaler("Mean IoU", epoch+1, mean_iou)



def initiate_training_environment(path: str):
    if not os.path.exists(path):
        os.mkdir(os.path.join(path))
    new_path = os.path.join(path, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.mkdir(new_path)
    os.mkdir(os.path.join(new_path, "weights"))
    os.mkdir(os.path.join(new_path, "samples"))

    return new_path



def main():
    configs = None
    with open("config.yaml", "r") as f:
        configs = yaml.safe_load(f)
    DEBUGGER_SINGLETON.active = configs["use_debugger"]
    
    train_path = initiate_training_environment(configs["output_dir"])
    configs["train_path"] = train_path
    LOG("configs", configs)
    
    train(configs=configs)


if __name__ == "__main__":
    main()
