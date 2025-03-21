import torch


def compute_iou(gen, gt, ths):
    IoUs = []
    for th in ths:
        gen = torch.sigmoid(gen)
        _volume = torch.ge(gen, th).float()
        intersection = torch.sum(_volume.mul(gt)).float()
        union = torch.sum(torch.ge(_volume.add(gt), 1)).float()
        iou = (intersection/union).item()
        IoUs.append(iou)

    return IoUs
