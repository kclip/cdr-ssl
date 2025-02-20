import torch
import torch.nn as nn


def _reduce(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "none":
        return tensor
    elif reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    else:
        raise ValueError(f"Unkown reduction '{reduction}'")


def _safe_div(num: torch.Tensor, den: torch.Tensor) -> torch.Tensor:
    fillna = torch.zeros_like(num)
    return torch.where(den == 0, fillna, num / den)


# Losses
# ------

class DetectionErrorLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predicted_logits, true_targets):
        predicted_targets = torch.argmax(predicted_logits, dim=-1, keepdim=True)
        detection_error = (predicted_targets.view(-1) != true_targets.view(-1)).type(torch.float)
        
        return _reduce(detection_error, self.reduction)


class AngleCosineLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, angles1, angles2):
        losses = 1 - torch.cos(angles2 - angles1)
        if losses.ndim > 1:
            losses = torch.mean(losses, dim=list(range(1, losses.ndim)))
        return _reduce(losses, reduction=self.reduction)
    

class MSESqueezeLoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, preds, targets):
        losses = torch.nn.functional.mse_loss(preds, targets, reduction=self.reduction)
        losses = losses.squeeze(dim=list(range(1, losses.ndim)))
        return losses


# Metrics
# -------

class AccuracyMetric(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, predicted_logits, true_targets):
        predicted_targets = torch.argmax(predicted_logits, dim=-1, keepdim=True)
        detection_error = (predicted_targets.view(-1) == true_targets.view(-1)).type(torch.float)
        
        return _reduce(detection_error, self.reduction)


class BinaryRecallMetric(nn.Module):    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # Recall
        tp = ((pred_t == 1) & (true_t == 1)).type(torch.float).sum()
        fn = ((pred_t == 0) & (true_t == 1)).type(torch.float).sum()
        recall = _safe_div(tp, (tp + fn))

        return recall


class MacroRecallMetric(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # Recall for each class
        recalls = []
        for c in range(self.n_classes):
            tp = ((pred_t == c) & (true_t == c)).type(torch.float).sum()
            fn = ((pred_t != c) & (true_t == c)).type(torch.float).sum()
            recalls.append(_safe_div(tp, (tp + fn)))
        
        # Average recalls across classes
        return torch.mean(torch.stack(recalls, dim=0))


class BinaryPrecisionMetric(nn.Module):    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # Precision
        tp = ((pred_t == 1) & (true_t == 1)).type(torch.float).sum()
        fp = ((pred_t == 1) & (true_t == 0)).type(torch.float).sum()
        precision = _safe_div(tp, (tp + fp))

        return precision


class MacroPrecisionMetric(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # Precision for each class
        precisions = []
        for c in range(self.n_classes):
            tp = ((pred_t == c) & (true_t == c)).type(torch.float).sum()
            fp = ((pred_t == c) & (true_t != c)).type(torch.float).sum()
            precisions.append(_safe_div(tp, (tp + fp)))
        
        # Average precisions across classes
        return torch.mean(torch.stack(precisions, dim=0))



class BinaryF1ScoreMetric(nn.Module):    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # F1-score
        tp = ((pred_t == 1) & (true_t == 1)).type(torch.float).sum()
        fp = ((pred_t == 1) & (true_t == 0)).type(torch.float).sum()
        fn = ((pred_t == 0) & (true_t == 1)).type(torch.float).sum()
        f1_score = _safe_div((2 * tp), ((2 * tp) + fp + fn))

        return f1_score


class MacroF1ScoreMetric(nn.Module):
    def __init__(self, n_classes: int = 2):
        super().__init__()
        self.n_classes = n_classes
    
    def forward(self, predicted_logits, true_targets):
        # Hard prediction
        pred_t = torch.argmax(predicted_logits, dim=-1, keepdim=True).view(-1)
        true_t = true_targets.view(-1)
        
        # F1-score for each class
        f1_scores = []
        for c in range(self.n_classes):
            tp = ((pred_t == c) & (true_t == c)).type(torch.float).sum()
            fp = ((pred_t == c) & (true_t != c)).type(torch.float).sum()
            fn = ((pred_t != c) & (true_t == c)).type(torch.float).sum()
            f1_scores.append(
                _safe_div((2 * tp), ((2 * tp) + fp + fn))
            )
        
        # Average f1-scores across classes
        return torch.mean(torch.stack(f1_scores, dim=0))
