import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss for binary classification """
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'): # gamma = 2.0 
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return F_loss

class ScriptoNetLoss(nn.Module):
    def __init__(self, lambda_det=1.0, lambda_cls=1.5, lambda_reg=0.5):
        super(ScriptoNetLoss, self).__init__()
        self.lambda_det = lambda_det # 
        self.lambda_cls = lambda_cls # 
        self.lambda_reg = lambda_reg # 
        
        self.detection_criterion = FocalLoss(gamma=2.0) # [cite: 123, 159]
        self.classification_criterion = nn.CrossEntropyLoss() # [cite: 124]
        self.regression_criterion = nn.MSELoss() # [cite: 125]

    def forward(self, det_logits, cls_logits, reg_score, det_target, cls_target, reg_target):
        # 1. Detection Loss (Focal Loss)
        loss_det = self.detection_criterion(det_logits.view(-1), det_target.float())
        
        # 2. Classification Loss (Cross Entropy)
        # Only compute for passages that actually contain a therapeutic event
        mask = (det_target == 1)
        if mask.sum() > 0:
            loss_cls = self.classification_criterion(cls_logits[mask], cls_target[mask])
            loss_reg = self.regression_criterion(reg_score[mask].view(-1), reg_target[mask].float())
        else:
            loss_cls = torch.tensor(0.0, device=det_logits.device, requires_grad=True)
            loss_reg = torch.tensor(0.0, device=det_logits.device, requires_grad=True)
            
        # Composite multi-task objective [cite: 121, 122]
        total_loss = (self.lambda_det * loss_det) + \
                     (self.lambda_cls * loss_cls) + \
                     (self.lambda_reg * loss_reg)
                     
        return total_loss, loss_det, loss_cls, loss_reg
