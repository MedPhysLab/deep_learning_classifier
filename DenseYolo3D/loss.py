import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0, reduce_th=0.0):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce_th = reduce_th
        self.eps = 1e-6

    def forward(self, input, target):
        y_pred = input.contiguous().view(-1)
        y_true = target.contiguous().view(-1)
        y_pred = torch.clamp(torch.sigmoid(y_pred), self.eps, 1.0)
        log_pt = -F.binary_cross_entropy(y_pred, y_true, reduction="none")
        pt = torch.exp(log_pt)
        if self.reduce_th > 0:
            th_pt = torch.where(
                pt < self.reduce_th,
                torch.ones_like(pt),
                ((1 - pt) / (1 - self.reduce_th)) ** self.gamma
            )
        else:
            th_pt = (1 - pt) ** self.gamma
        loss = -self.alpha * th_pt * log_pt
        return torch.sum(loss) / (y_true.numel() + self.eps)  # Normalize by total cells

class LocalizationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(LocalizationLoss, self).__init__()
        self.weight = weight
        self.eps = 1e-6

    def forward(self, input, target):
        """
        Args:
            input: [batch_size, 6, grid_d, grid_h, grid_w] (x, y, z, w, h, d)
            target: [batch_size, 7, grid_d, grid_h, grid_w] (confidence, x, y, z, w, h, d)
        Returns:
            loss: Scalar, sum of offset and box size losses
        """
        mask = (target[:, 0, :, :, :] == 1.0).float()  # [batch_size, grid_d, grid_h, grid_w]
        
        # Offset loss (x, y, z)
        input_offset = input[:, 0:3, :, :, :] * mask.unsqueeze(1)  # [batch_size, 3, grid_d, grid_h, grid_w]
        target_offset = target[:, 1:4, :, :, :] * mask.unsqueeze(1)
        loss_offset = F.mse_loss(
            input_offset.contiguous().view(-1),
            target_offset.contiguous().view(-1),
            reduction="sum"
        ) / (torch.sum(mask) + self.eps)
        
        # Box size loss (w, h, d)
        input_box = input[:, 3:6, :, :, :] * mask.unsqueeze(1)  # [batch_size, 3, grid_d, grid_h, grid_w]
        target_box = target[:, 4:7, :, :, :] * mask.unsqueeze(1)
        # Zero predictions where target is zero to avoid penalizing undefined sizes
        input_box = torch.where(
            target_box == 0.0,
            torch.zeros_like(input_box),
            input_box
        )
        loss_box = F.mse_loss(
            input_box.contiguous().view(-1),
            target_box.contiguous().view(-1),
            reduction="sum"
        ) / (torch.sum(mask) + self.eps)
        
        return self.weight * (loss_offset + loss_box)

class YOLO3DLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1.0, lambda_coord=5.0):
        super(YOLO3DLoss, self).__init__()
        self.obj_loss = BinaryFocalLoss(alpha=alpha, gamma=gamma, reduce_th=0.0)  # Standard focal loss
        self.loc_loss = LocalizationLoss(weight=lambda_coord)

    def forward(self, pred, target):
        """
        Args:
            pred: [batch_size, 7, grid_d, grid_h, grid_w]
                - pred[:, 0, :, :, :]: Objectness confidence (logits)
                - pred[:, 1:4, :, :, :]: Location (x, y, z)
                - pred[:, 4:7, :, :, :]: Box size (w, h, d)
            target: [batch_size, 7, grid_d, grid_h, grid_w]
                - target[:, 0, :, :, :]: Ground truth confidence
                - target[:, 1:7, :, :, :]: Ground truth bbox (x, y, z, w, h, d)
        Returns:
            loss: Scalar
            components: Dict with obj_loss, loc_loss
        """
        # Objectness loss
        obj_pred = pred[:, 0, :, :, :]  # [batch_size, grid_d, grid_h, grid_w]
        obj_target = target[:, 0, :, :, :]  # [batch_size, grid_d, grid_h, grid_w]
        obj_loss = self.obj_loss(obj_pred, obj_target)
        
        # Localization loss
        box_pred = pred[:, 1:7, :, :, :]  # [batch_size, 6, grid_d, grid_h, grid_w]
        loc_loss = self.loc_loss(box_pred, target)
        
        # Total loss
        total_loss = obj_loss + loc_loss
        
        components = {
            "obj_loss": obj_loss.item(),
            "loc_loss": loc_loss.item()
        }
        return total_loss, components









# Example usage
if __name__ == "__main__":
    # Create dummy data
    pred = torch.randn(2, 7, 4, 4, 4)  # [batch_size, 7, grid_d, grid_h, grid_w]
    target = torch.randn(2, 7, 4, 4, 4)  # [batch_size, 7, grid_d, grid_h, grid_w]
    
    # Create loss function
    loss_fn = YOLO3DLoss()
    
    # Compute loss
    loss, components = loss_fn(pred, target)
    
    print("Total Loss:", loss.item())
    print("Components:", components)






