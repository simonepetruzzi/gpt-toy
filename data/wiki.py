import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from typing import List

class WikiDataset:
    def __init__(self, root, tokens: List[int], block_size: int):
        """
        Dataset for tokenized Wikipedia data.
        
        Args:
            tokens (List[int]): List of token IDs.
            block_size (int): Size of each block of tokens.
        """
        self.tokens = tokens
        self.block_size = block_size

        def __len__(self):
            return len(self.tokens) // self.block_size

        def __getitem__(self, idx):
            start_idx = idx * self.block_size
            end_idx = start_idx + self.block_size
            return torch.tensor(self.tokens[start_idx:end_idx], dtype=torch.long)



        def load_tokens(file_path):
            """Loads tokenized text from a file."""
            with open(file_path, "r", encoding="utf-8") as f:
                tokens = f.read().split()
                return list(map(int, tokens))


        def prepare_datasets(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
            """
            Prepare datasets and dataloaders for training, validation, and testing.

            Args:
                cfg (DictConfig): Hydra configuration object.

            Returns:
                dict: Dictionary containing DataLoaders for train, val, and test sets.
            """
            data_dir = cfg.data_dir
            block_size = cfg.preprocessing.block_size

            # Load tokenized data
            train_tokens = load_tokens(os.path.join(data_dir, "wiki.train.tokens"))
            valid_tokens = load_tokens(os.path.join(data_dir, "wiki.valid.tokens"))
            test_tokens = load_tokens(os.path.join(data_dir, "wiki.test.tokens"))

            # Create datasets
            train_dataset = WikiDataset(train_tokens, block_size)
            val_dataset = WikiDataset(valid_tokens, block_size)
            test_dataset = WikiDataset(test_tokens, block_size)

            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.batch_size,
                shuffle=cfg.shuffle,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=cfg.batch_size,
                shuffle=False,
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory
            )

            return train_loader, val_loader, test_loader
