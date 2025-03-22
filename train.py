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
    IoUs = [[] for _ in range(len(THRESHOLDS))]
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            images, volumes = batch
            images = images.to(configs["device"])
            volumes = volumes.to(configs["device"])

            outputs = model(images)

            loss = loss_fn(outputs, volumes)
            VAL_LOSS_ACCUMULATOR+=loss.item()
            _IOUs = compute_iou(outputs, volumes, ths=THRESHOLDS)  ## ASSUMING THEY ARE 3
            for i, IoU in enumerate(_IOUs):
                IoUs[i].append(IoU)

        mean_val_loss = VAL_LOSS_ACCUMULATOR/ITERATIONS_PER_EPOCH
        mean_IoUs = [sum(ious)/len(ious) for ious in IoUs]

    return mean_val_loss, mean_IoUs

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



    model = full_model.Pix2VoxSharp(configs).to(configs["device"])
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=trainable_params, lr=configs["optim"]["lr"])

    if not configs["train"]["continue_from_checkpoint"]:
        START_EPOCH = 0
        current_best_mAP = 0
    else: 
        print("loading checkpoint")
        START_EPOCH, current_best_mAP, model_state_dict, optimizer_state_dict = network_utils.load_checkpoint(configs)
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)
        print("state dict loaded without any problems")

    
    THRESHOLDS = configs["thresholds"]
    EPOCHS = configs["train"]["epochs"]
    loss_fn = VoxelLoss(weight=10)
    ITERATIONS_PER_EPOCH_TRAIN = int(len(train_dataset)/BATCH_SIZE)
    ITERATIONS_PER_EPOCH_VAL = int(len(val_dataset)/BATCH_SIZE)


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
        valid_loss, valid_IoUs = compute_validation_metrics(model, loss_fn, val_loader, THRESHOLDS, ITERATIONS_PER_EPOCH_VAL, configs)
        mAP = sum(valid_IoUs)/len(valid_IoUs)

        LOG("average train loss", average_epoch_loss)
        LOG("average test loss", valid_loss)
        LOG("test IoU @ different THs", valid_IoUs)
        for TH, iou in zip(THRESHOLDS, valid_IoUs):
            LOG(f"iou @ {TH}: {iou}")
        LOG(f"mAP@{THRESHOLDS[0]}-{THRESHOLDS[-1]}", mAP)
        


        if mAP > current_best_mAP:
            current_best_mAP = mAP
            CHECKPOINT(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")
            writer.add_line(f"IoU has scored a higher value at epoch {epoch+1}. Saving Weights...")
            weights_path = os.path.join(configs["train_path"], "weights", "best.pth")
            network_utils.save_checkpoints(weights_path, epoch+1,model, optimizer, mAP, epoch+1)
            volumes_path = os.path.join(configs["train_path"], "samples", f"output{epoch+1}.pth")
            images_path = os.path.join(configs["train_path"], "samples", f"images{epoch+1}.pth")

            torch.save(outputs, volumes_path)
            torch.save(images, images_path)
            LOG("tensor saved")
            
        if (epoch+1) % configs["train"]["save_every"] == 0:
            weights_path = os.path.join(configs["train_path"], "weights", "last.pth")
            CHECKPOINT("Saving last Weights...")
            network_utils.save_checkpoints(weights_path, epoch+1,model, optimizer, mAP, epoch+1)

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
        for TH, iou in zip(THRESHOLDS, valid_IoUs):
            writer.add_line(f"IoU @ {TH}: {iou}")

        writer.add_scaler("Mean IoU", epoch+1, mAP)



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
    train(configs=configs)


if __name__ == "__main__":
    main()
