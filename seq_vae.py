import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np 
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd

from seq_vae_experiments_tool import *
from seq_datamodule import DataModule
from utils import standardize_data, preprocess_data
pl.seed_everything(123)

class RNNEncoder(nn.Module):
    def __init__(self,
                 input_size: int=3,
                 hidden_size: int=10,
                 rnn_layers: int=1,
                 *args, **kwargs) -> None:
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, batch_first=True, 
                          hidden_size=hidden_size, num_layers=rnn_layers).float()
        self.hidden_size = hidden_size

    def forward(self, x):
        x, rnn_hidden_states = self.rnn(x)
        # reshape to (batch_size, seq_len*input_size) to be consistent with linear encoder
        hidden_state = rnn_hidden_states[-1, :, :]
        return hidden_state

class RNNDecoder(nn.Module):
    def __init__(self, 
                 latent_dim=4, 
                 hidden_size=10,
                 rnn_layers=4, 
                 input_size=3, 
                 seq_len=50,
                 *args, **kwargs):
        super(RNNDecoder, self).__init__()
        self.rnn1 = nn.GRU(input_size=latent_dim, batch_first=True,
                           hidden_size=hidden_size, num_layers=rnn_layers).float()
        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.seq_len=seq_len

    def forward(self, x):
        x = x.reshape(-1, 1, x.shape[-1]).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x_hat = self.fc(x)
        return x_hat
    
class MLPEncoder(nn.Module):
    def __init__(self, input_size=100, 
                 seq_len=50, 
                 hidden_size=256, 
                 latent_size=10, 
                 *args, **kwargs):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(seq_len*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, latent_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_in = x.reshape(x.shape[0], -1)
        out = self.relu(self.fc1(x_in))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        return out
    
class MLPDecoder(nn.Module):
    def __init__(self, latent_size=1, 
                 hidden_size=256,
                 seq_len=50,
                 input_size=100, 
                 *args, **kwargs):
        super(MLPDecoder, self).__init__()
        self.input_size=input_size
        self.seq_len=seq_len
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_len*input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out.reshape(-1, self.seq_len, self.input_size)
    

class seq_vae(pl.LightningModule):
    def __init__(self, enc_out_dim, learning_rate, 
                 latent_dim, input_size, 
                 seq_len, beta, 
                 hidden_size, rnn_layers,
                 rnn_models,
                 *args, **kwargs):
        super(seq_vae, self).__init__()
        self.save_hyperparameters()

        self.beta = beta
        self.learning_rate = learning_rate

        if rnn_models:
            self.encoder = RNNEncoder(input_size=input_size, hidden_size=hidden_size, rnn_layers=rnn_layers)
            self.decoder = RNNDecoder(latent_dim=latent_dim, hidden_size=hidden_size, rnn_layers=rnn_layers, 
                                      input_size=input_size, seq_len=seq_len)
        else:
            self.encoder = MLPEncoder(input_size=input_size, hidden_size=hidden_size, seq_len=seq_len)
            self.decoder = MLPDecoder(latent_size=latent_dim, hidden_size=hidden_size, 
                                      input_size=input_size, seq_len=seq_len)
            
        self.fc_mu = nn.Linear(hidden_size, latent_dim)
        self.fc_var = nn.Linear(hidden_size, latent_dim)
        self.log_scale_diag = nn.Parameter(torch.zeros(seq_len * input_size))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def gaussian_likelihood(self, x_hatt, logscale_diag, x):
        scale_diag = torch.exp(logscale_diag)
        scale = torch.diag(scale_diag)
        mu_x = x_hatt.reshape(x_hatt.shape[0], -1)
        dist = torch.distributions.MultivariateNormal(mu_x, scale_tril=scale)

        log_pxz = dist.log_prob(x.reshape(mu_x.shape[0], -1))
        return log_pxz
    
    def kl_divergence(self, mu, std):
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        kl = torch.distributions.kl.kl_divergence(q, p)
        kl = kl.sum(-1)
        return kl
    
    def forward(self, x):
        x_enc = self.encoder(x)
        mu, log_var = self.fc_mu(x_enc), self.fc_var(x_enc)

        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu, std)
        z = q.sample()

        x_hat = self.decoder(z)
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale_diag, x)
        kl = self.kl_divergence(mu, std)
        
        elbo = (self.beta * kl - recon_loss)
        elbo = elbo.mean()

        return elbo, {'elbo':elbo, 'kl':kl, 'recon_loss':recon_loss}
    
    def shared_eval(self, x):
        x_enc = self.encoder(x)
        mu, log_var = self.fc_mu(x_enc), self.fc_var(x_enc)

        std = torch.exp(log_var/2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        x_hatt = self.decoder(z)

        recon_loss = self.gaussian_likelihood(x_hatt, self.log_scale_diag, x)
        recon_loss_mean = torch.mean(recon_loss)
        kl = self.kl_divergence(mu, std)
        kl_mean = torch.mean(kl)
        elbo = (self.beta * kl - recon_loss)
        elbo = elbo.mean()
    
        return elbo, {'elbo':elbo, 'kl':kl_mean, 'recon_loss':recon_loss_mean}
    
    def training_step(self, batch, batch_idx):
        x = batch
        train_elbo, train_log_dict = self.shared_eval(x)
        self.log('Loss', train_log_dict['elbo'])
        self.log('kl', train_log_dict['kl'])
        self.log('recon_loss', train_log_dict['recon_loss'])
        return train_log_dict['elbo']

    def validation_step(self, batch, batch_idx):
        x = batch
        train_elbo, train_log_dict = self.shared_eval(x)
        self.log('val_loss', train_log_dict['elbo'])
        self.log('kl', train_log_dict['kl'])
        self.log('recon_loss', train_log_dict['recon_loss'])
        return train_log_dict['elbo']
    
    def test_step(self, batch, batch_idx):
        x = batch
        train_elbo, train_log_dict = self.shared_eval(x)
        self.log('Loss', train_log_dict['elbo'])
        self.log('kl', train_log_dict['kl'])
        self.log('recon_loss', train_log_dict['recon_loss'])
        return train_log_dict['elbo']
        

def train(hparams:dict):

    logger = TensorBoardLogger('lightning_logs', name='seq_vae_tank_v9', default_hp_metric=False)
    np.random.seed(123)
    torch.manual_seed(123)
    model = seq_vae(**hparams)

    # tank data
    df = pd.read_csv(f'preprocessed_data/tank_simulation/norm_long.csv').reset_index(drop=True).iloc[1000:, :3]

    # BeRfiPl data
    #_, data, _, _, _ = preprocess_data('BeRfiPl_labels')
    #df = pd.DataFrame(data)

    # Siemens data
    #df = pd.read_csv('preprocessed_data/SmA/id1_norm.csv').drop(columns=['CuStepNo ValueY']).iloc[8000:30000, :].reset_index(drop=True)
    #df_train, df_val = train_test_split(df, test_size=0.2)

    df_sc = pd.DataFrame(standardize_data(df, f'scaler_dataset_train.pkl'), columns=df.columns)

    data_module = DataModule(**hparams, dataframe=df_sc)    
    early_stop_callback = EarlyStopping(monitor="val_loss", mode='min', patience=30)
    trainer = pl.Trainer(max_epochs=500, log_every_n_steps=1, logger=logger, accelerator='gpu', devices=1, callbacks=[early_stop_callback])
    trainer.fit(model, data_module)


if __name__ == '__main__':
    # train()
    # complete code execution by use of the experiments method to compute several experiments after each other
    experiments_grid = make_grid()
    experiments_to_json(experiments_grid)
    experiments = load_experiments(modus='run')
    for experiment in range(len(experiments)):
        print('experiment no.', experiment)
        train(hparams=experiments[str(experiment)])
        print('completed experiment no ' + str(experiment + 1) + str(len(experiments)) + ' experiments')

