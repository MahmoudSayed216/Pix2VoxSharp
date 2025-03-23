import torch.nn as nn
import torch.nn.init as init
import torch
from datetime import datetime as dt
from utils.debugger import CHECKPOINT
from collections import OrderedDict
import os


def init_weights(l):
    if  (isinstance(l, nn.Conv2d) or isinstance(l, nn.Conv3d) or isinstance(l, nn.ConvTranspose3d)):
        init.kaiming_normal_(l.weight)
        if l.bias is not None:
            init.constant_(l.bias, 0)

    elif (isinstance(l, nn.BatchNorm2d) or isinstance(l, nn.BatchNorm3d)):
        init.constant_(l.weight, 1)
        init.constant_(l.bias, 0)

    elif (isinstance(l, nn.Linear)):
        init.normal_(l.weight, 0, 0.01)
        init.constant_(l.bias, 0)

def save_checkpoints(file_path, epoch_idx, model, model_solver, iou, epoch):
    print('[INFO] %s Saving checkpoint to %s ...' % (dt.now(), file_path))
    CHECKPOINT(f"{dt.now()} Saving checkpoints to {file_path}")
    checkpoint = {
        'epoch_idx': epoch_idx,
        'iou': iou,
        'best_epoch': epoch,
        'model_state_dict': model.state_dict(),
        'model_solver_state_dict': model_solver.state_dict(),
    }

    torch.save(checkpoint, file_path)


def load_checkpoint(configs):
    checkpoint_id = configs["train"]["checkpoint_id"]
    output_path = configs["output_dir"]
    checkpoint_type = configs["train"]["checkpoint_type"]
    path = os.path.join("/kaggle/input/2025-03-22_23-40-57/pytorch/default/1/best.pth")

    checkpoint:dict = torch.load(path, map_location=configs["device"])

    print(checkpoint.keys())
    epoch = checkpoint['epoch_idx']
    iou = checkpoint['iou']
    model_state_dict:OrderedDict = checkpoint['model_state_dict']
    solver_state_dict = checkpoint['model_solver_state_dict']


    return epoch, iou, model_state_dict, solver_state_dict
