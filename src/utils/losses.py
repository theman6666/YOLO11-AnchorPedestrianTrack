import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for classification (binary or multi-label)
    gamma: focusing parameter
    alpha: balance
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: logits or probabilities? we assume logits
        prob = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = prob * targets + (1 - prob) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_factor = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_factor * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# SIoU: an example implementation (paper: SIoU loss)
# This is a direct implementation adapted for bbox tensors in form (x,y,w,h) or (x1,y1,x2,y2)
# For integration into ultralytics, one usually needs to replace the CIoU calculation with this function
def siou_loss(pred_boxes, target_boxes, reduction='mean'):
    """
    pred_boxes, target_boxes: tensors of shape (N,4) in xywh or xyxy? We use xyxy here.
    Returns 1 - siou
    """
    # Ensure xyxy
    px1, py1, px2, py2 = pred_boxes[:,0], pred_boxes[:,1], pred_boxes[:,2], pred_boxes[:,3]
    gx1, gy1, gx2, gy2 = target_boxes[:,0], target_boxes[:,1], target_boxes[:,2], target_boxes[:,3]

    # intersection
    ix1 = torch.max(px1, gx1)
    iy1 = torch.max(py1, gy1)
    ix2 = torch.min(px2, gx2)
    iy2 = torch.min(py2, gy2)

    iw = (ix2 - ix1).clamp(min=0)
    ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    # areas
    area_p = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
    area_g = (gx2 - gx1).clamp(min=0) * (gy2 - gy1).clamp(min=0)
    union = area_p + area_g - inter + 1e-7

    iou = inter / union

    # center distance
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    gcx = (gx1 + gx2) / 2
    gcy = (gy1 + gy2) / 2

    dx = gcx - pcx
    dy = gcy - pcy
    rho2 = dx**2 + dy**2

    # enclose box
    cx1 = torch.min(px1, gx1)
    cy1 = torch.min(py1, gy1)
    cx2 = torch.max(px2, gx2)
    cy2 = torch.max(py2, gy2)
    cw = (cx2 - cx1)
    ch = (cy2 - cy1)
    c2 = (cw**2 + ch**2) + 1e-7

    # angle and shape cost (SIoU core ideas simplified)
    # sigma = rho2 / c2
    # add angle / shape terms (simplified)
    v = (4 / (3.1415926 ** 2)) * (torch.atan((gx2-gx1)/(gy2-gy1)+1e-7) - torch.atan((px2-px1)/(py2-py1)+1e-7))**2
    alpha = v / (1 - iou + v + 1e-7)
    # siou
    siou = iou - (rho2 / c2) - alpha * v
    loss = 1 - siou
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss
