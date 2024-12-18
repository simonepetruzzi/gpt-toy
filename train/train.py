import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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


try:
    import wandb
except ImportError:
    print('Wandb is not installed. If you want to use it, please install it with pip install wandb')
    wandb = None


@hydra.main(version_base=None, config_path="../configs", config_name="train_config")

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

    # Load the dataset
    dataset = OpenWebTextDataset(cfg.dataset.data_dir, cfg.dataset.block_size)
    train_loader = DataLoader(dataset.train_dataset, batch_size=cfg.training.train_batch_size, shuffle=True, num_workers=cfg.training.num_workers)
    val_loader = DataLoader(dataset.val_dataset, batch_size=cfg.training.eval_batch_size, shuffle=False, num_workers=cfg.training.num_workers)

    # Initialize model
    model = GPT(cfg.model.n_layer, cfg.model.n_head, cfg.model.n_embd, cfg.model.vocab_size, cfg.model.dropout)
    model.to(device)
