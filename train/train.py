import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision import transforms as t
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchmetrics
import hydra
from omegaconf import OmegaConf, DictConfig
from hydra.utils import instantiate
from pprint import pprint
from torch.utils.data import Subset

from utils.utils import save_checkpoint

try:
    import wandb
except ImportError:
    print('Wandb is not installed. If you want to use it, please install it with pip install wandb')
    wandb = None

@hydra.main(version_base=None, config_path="../configs", config_name="train_config.yaml")

def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # Experiment name and settings
    exp_name = cfg.experiment_name
    device = torch.device(cfg.device)
    experiment_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    experiment_dir, checkpoints_dir = make_experiment_directory(experiment_dir)

    # Logger
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(config_dict)
    logger = instantiate(cfg.logger, settings=str(config_dict), dir=experiment_dir)

    # Initialize dataset and dataloader
    training_args = cfg.training
    dataset = instantiate(cfg.dataset)
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader
    test_loader = dataset.test_loader
    
     # Initialize model
    model = instantiate(cfg.model).to(device)

    #Load checkpoints (to do)
    
    # Main loss
    main_loss = instantiate(cfg.loss.classification_loss)

    # Additional loss (to do)

    # Metrics
    metric = torchmetrics.Perplexity()

    # Optimizer and scheduler
    optimizer = instantiate(cfg.optimizer, params = model.parameters())
    scheduler = None  
    if 'scheduler' in cfg:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # Training loop
    for epoch in range(training_args.num_epochs):
        model.train()
        total_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{training_args.num_epochs}"):
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs, labels=labels)
            logits = outputs.logits

            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = main_loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            print(f"Batch {i}/{len(train_loader)}: Loss = {loss.item():.4f}")

            # Backpropagation
            loss.backward()
            clip_grad_norm_(model.parameters(), training_args.max_grad_norm)
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            # Update metrics with logits and labels, not the loss
            metric.update(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        # Log metrics
        avg_loss = total_loss / len(train_loader)
        avg_perplexity = metric.compute()

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Perplexity = {avg_perplexity:.4f}")

        # Log metrics to logger
        if wandb:
            wandb.log({"epoch": epoch + 1, "loss": avg_loss, "perplexity": avg_perplexity})

        # Save checkpoint if validation loss improves
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(model, optimizer, epoch + 1, checkpoints_dir, scheduler=scheduler, best_loss=best_loss)

        # Reset perplexity metric for the next epoch
        metric.reset()



if __name__ == '__main__':
     train()
