import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import Dataset, DataLoader, random_split
import torch
from utils import standardize_data
import pytorch_lightning as pl

class Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, number_timesteps, *args, **kwargs):
        len_dataset = len(dataframe)
        number_timesteps = number_timesteps
        len_last_window = len_dataset-number_timesteps

        self.df = torch.tensor(dataframe.values.astype(np.float32))
        self.start_idx_list = [i for i in range(len_last_window)]
        self.end_idx_list = [i + number_timesteps for i in self.start_idx_list]

    def __len__(self):
        return len(self.start_idx_list)
    
    def __getitem__(self, index):
        return self.df[self.start_idx_list[index]: self.end_idx_list[index], :]
    


class DataModule(pl.LightningDataModule):
    def __init__(self, dataframe, valid_split, batch_size, num_workers, number_timesteps, *args, **kwargs):
        self.valid_split = valid_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataframe = dataframe
        self.number_timestemps = number_timesteps
        super().__init__()

    def setup(self, stage=None):
        dataset = Dataset(self.dataframe, number_timesteps=self.number_timestemps)
        data_shape = len(dataset)
        valid_len = int(np.floor(data_shape * self.valid_split))
        train_len = data_shape - valid_len
        self.dataset_train, self.dataset_valid = random_split(dataset=dataset, lengths=[train_len, valid_len], generator=torch.Generator())

    def train_dataloader(self):
        return DataLoader(self.dataset_train, 
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.dataset_valid, 
                          shuffle=False,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

