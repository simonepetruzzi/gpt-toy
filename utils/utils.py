import os
import torch

def save_checkpoint(model, optimizer, epoch, save_dir, scheduler=None, best_loss=None):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    if scheduler:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if best_loss is not None:
        checkpoint["best_loss"] = best_loss

    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth")
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch} in {checkpoint_path}")
