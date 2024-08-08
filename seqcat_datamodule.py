import numpy as np
import pandas as pd
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS 
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pytorch_lightning as pl
from utils import standardize_data, standardize_seq


class Dataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame,
                  number_timesteps, 
                  *args, **kwargs):
        len_dataset = len(dataframe)
        len_last_window = len_dataset - number_timesteps
        self.df = torch.tensor(dataframe.values.astype(np.float32))
        self.start_idx_list = [i for i in range(len_last_window)]
        self.end_idx_list = [i + number_timesteps for i in self.start_idx_list]

    def __len__(self):
        return len(self.start_idx_list)
    
    def __getitem__(self, index):
        return self.df[self.start_idx_list[index]: self.end_idx_list[index], :]

    
class DataModule(pl.LightningDataModule):
    def __init__(self, dataframe, hparams: dict, 
                 *args, **kwargs):
        self.valid_split = hparams['VALID_SPLIT']
        self.batch_size = hparams['BATCH_SIZE']
        self.num_workers = hparams['NUM_WORKERS']
        self.dataframe = dataframe
        self.number_timesteps = hparams['NUMBER_TIMESTEPS']
        super().__init__()

    def setup(self, stage=None):
        dataset = Dataset(dataframe=self.dataframe, number_timesteps=self.number_timesteps)
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
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.dataset_valid, 
                          shuffle=False, 
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)
    
if __name__ == '__main__':
    hparams = {"NUMBER_TIMESTEPS" : 10,
                    "SEQ_LEN" : 10,
                    "VALID_SPLIT" : 0.2,
                    "BETA" : 0.1,
                    "TEMP" : 0.5,
                    "BATCH_SIZE" : 50,
                    "NUM_WORKERS" : 20,
                    "CATEGORICAL_DIM" : 5,
                    "ENC_OUT_DIM" : 5,
                    "IN_DIM" : 154,
                    "HIDDEN_DIM" : 10,
                    "RNN_MODELS" : True,
                    "RNN_LAYERS" : 1}

    df = pd.read_csv('preprocessed_data/BeRfiPl/ds1n.csv', index_col=0).reset_index(drop=True)
    data_mod = DataModule(hparams=hparams, dataframe=df)
    data_mod.setup()
    dl = data_mod.train_dataloader()
    train_batch =next(iter(dl))
    breakpoint()
