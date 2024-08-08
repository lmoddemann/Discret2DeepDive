import torch 
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import numpy as np
import pandas as pd
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from seqcat_experiments_tool import *
from seqcat_datamodule import DataModule
from utils import preprocess_data, preprocess_data_artificial, preprocess_data_siemens, preprocess_data_simu_tank
from utils import standardize_data
pl.seed_everything(123)

class RNNEncoder(nn.Module):
    def __init__(self, 
                 hparams:dict,
                 *args, **kwargs):
        super(RNNEncoder, self).__init__()

        self.categorical_dim= hparams['CATEGORICAL_DIM']
        self.enc_out_dim = hparams['ENC_OUT_DIM']
        self.input_size = hparams['IN_DIM']
        self.rnn_layers = hparams['RNN_LAYERS']
        self.rnn = nn.GRU(input_size=self.input_size, batch_first=True, 
                          hidden_size=self.enc_out_dim, num_layers=self.rnn_layers).float()

    def forward(self, x):
        x, rnn_hidden_states = self.rnn(x)
        hidden_state = rnn_hidden_states[-1, :, :]
        return hidden_state
    
class RNNDecoder(nn.Module):
    def __init__(self, 
                 hparams: dict, 
                 *args, **kwargs):
        super(RNNDecoder, self).__init__()

        self.categorical_dim = hparams['CATEGORICAL_DIM']
        self.hidden_size = hparams['HIDDEN_DIM']
        self.rnn_layers = hparams['RNN_LAYERS']
        self.input_size = hparams['IN_DIM']
        self.seq_len = hparams['SEQ_LEN']
        self.rnn1 = nn.GRU(input_size=self.categorical_dim, batch_first=True,
                           hidden_size=self.hidden_size, num_layers=self.rnn_layers).float()
        self.fc = nn.Linear(in_features=self.hidden_size, out_features=self.input_size)

    def forward(self, x): 
        x = x.reshape(-1, 1, x.shape[-1]).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        x_hat = self.fc(x)
        return x_hat
    
class MLPEncoder(nn.Module): 
    def __init__(self, hparams:dict, 
                 *args, **kwargs):
        super(MLPEncoder, self).__init__()

        self.categorical_dim = hparams['CATEGORICAL_DIM']
        self.hidden_size = hparams['HIDDEN_DIM']
        self.enc_out_dim = hparams['ENC_OUT_DIM']
        self.rnn_layers = hparams['RNN_LAYERS']
        self.input_size = hparams['IN_DIM']
        self.seq_len = hparams['SEQ_LEN']

        self.fc1 = nn.Linear(self.seq_len * self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.enc_out_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()


    def forward(self, x): 
        x_in = x.reshape(x.shape[0], -1)
        out = self.relu(self.fc1(x_in))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class MLPDecoder(nn.Module):
    def __init__(self, hparams:dict,  
                 *args, **kwargs):
        super(MLPDecoder, self).__init__()

        self.categorical_dim = hparams['CATEGORICAL_DIM']
        self.seq_len = hparams['SEQ_LEN']
        self.hidden_size = hparams['HIDDEN_DIM']
        self.input_size = hparams['IN_DIM']

        self.fc1 = nn.Linear(self.categorical_dim, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.seq_len * self.input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out.reshape(-1, self.seq_len, self.input_size)
    
class seqcat_vae(pl.LightningModule):
    def __init__(self, hparams:dict,
                 *args, **kwargs):
        super(seqcat_vae, self).__init__()
        self.save_hyperparameters()

        self.beta = hparams["BETA"]
        self.categorical_dim = hparams['CATEGORICAL_DIM']
        self.input_size = hparams['IN_DIM']
        self.rnn_models = hparams['RNN_MODELS']
        self.hidden_size = hparams['HIDDEN_DIM']
        self.rnn_layers = hparams['RNN_LAYERS']
        self.seq_len = hparams['SEQ_LEN']
        self.enc_out_dim = hparams['ENC_OUT_DIM']
        self.temp = hparams['TEMP']

        if self.rnn_models:
            self.encoder = RNNEncoder(hparams=hparams)
            self.decoder = RNNDecoder(hparams=hparams)
            
        else: 
            self.encoder = MLPEncoder(hparams=hparams)
            self.decoder = MLPDecoder(hparams=hparams)

        # Encoder hidden to latent categorical distribution
        self.fc_z_cat = nn.Linear(self.enc_out_dim, self.categorical_dim)
        self.fc_mu_x = nn.Linear(self.input_size, self.input_size)
        self.fc_logvar_x = nn.Linear(self.input_size, self.input_size)

        # For the gaussian likelihood
        self.log_scale_diag = nn.Parameter(torch.zeros(self.seq_len * self.input_size))
        # prior distribution
        self.pz = torch.distributions.OneHotCategorical(
            1. / self.categorical_dim * torch.ones(1, self.categorical_dim, device='cuda'))
        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.)[0], requires_grad=True )


    def encode(self, input: torch.Tensor) -> torch.Tensor:
        result = self.encoder(input)
        z = self.fc_z_cat(torch.flatten(result, start_dim=1))
        z_out = z.view(-1, self.categorical_dim)
        return z_out
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result = self.decoder(z)
        mu = self.fc_mu_x(result)
        logvar = self.fc_logvar_x(result)
        sigma = torch.diag_embed(torch.exp(logvar))
        return mu, sigma

    def sample_gumble(self, logits: torch.Tensor, eps: float=1e-7) -> torch.Tensor:
        u = torch.rand_like(logits)
        g = - torch.log(- torch.log(u + eps) + eps)
        s = F.softmax((logits + g) / self.temp, dim = -1)
        return s
    
    def gaussian_likelihood(self, x_hat, logscale_diag, x):
        scale_diag = torch.exp(logscale_diag)
        scale = torch.diag(scale_diag)
        mu_x = x_hat.reshape(x_hat.shape[0], -1)
        dist = torch.distributions.MultivariateNormal(mu_x, scale_tril=scale)
        likelihood = dist.log_prob(x.reshape(mu_x.shape[0], -1))
        mean_likelihood = torch.mean(likelihood)
        return likelihood, mean_likelihood
    
    def kl_divergence(self, pzx):
        kl_categorical = torch.distributions.kl.kl_divergence(pzx, self.pz)
        mean_kl = torch.mean(kl_categorical)
        return kl_categorical, mean_kl
    
    def shared_eval(self, x):
        # computing parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categorical dist. object for use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx
        z = self.sample_gumble(logits=pzx_logits)
        # decode into mu and sigma
        mu, sigma = self.decode(z)
        # construct multivariate distributional pbject for pxz
        pxz = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z

    def forward(self, x: torch.Tensor, **kwargs):
        return self.shared_eval(x)

    def loss_function(self, mu, x, pzx):
        likelihood, mean_likelihood = self.gaussian_likelihood(x_hat = mu, logscale_diag=self.log_scale_diag, x=x)
        kl_categorical, mean_kl = self.kl_divergence(pzx=pzx)
        elbo = self.beta * kl_categorical - likelihood
        elbo = elbo.mean()
        return {'Loss': elbo, 'likelihood': mean_likelihood, 'KLD_cat': mean_kl}
    
    def get_states(self, x):
        # computung parameters of categorial dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categoricl dist. object for use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx
        z = self.sample_gumble(logits=pzx_logits)
        # compute states by using the argmax of logits
        z_states = torch.zeros(z.shape).to(device='cuda').scatter(1, torch.argmax(pzx_logits, dim=1).unsqueeze(1), 1)
        # decode into mu and sigma
        mu, sigma = self.decode(z_states)
        # construct mutlicariate distribution object for pxz
        pxz = torch.distributions.MultivariateNormal(
                    loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z_states

    def function_likelihood(self, x):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        likelihood = pxz.log_prob(x)
        return likelihood
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def training_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct = self.loss_function(mu, x, pzx)
        self.log('Loss', loss_dct['Loss'])
        self.log('likelihood', loss_dct['likelihood'])
        self.log('KLD_cat', loss_dct['KLD_cat'])
        self.log('train_loss', loss_dct['Loss'])
        return loss_dct['Loss']
    
    def validation_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct = self.loss_function(mu, x, pzx)
        self.log('val_loss', loss_dct['Loss'])
        self.log('likelihood', loss_dct['likelihood'])
        self.log('KLD_cat', loss_dct['KLD_cat'])
        return loss_dct['Loss']
    
    def test_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct = self.loss_function(mu, x, pzx)
        self.log('Loss', loss_dct['Loss'])
        self.log('likelihood', loss_dct['likelihood'])
        self.log('KLD_cat', loss_dct['KLD_cat'])
        return loss_dct['Loss']


def train(hparams:dict):
    logger = TensorBoardLogger('lightning_logs', name='seqcat_berfipl_RNN_v25', default_hp_metric=False)  
   
    import random
    random_seed = random.randint(0, 10000)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model = seqcat_vae(hparams=hparams)

    # tank data
    #df = pd.read_csv(f'preprocessed_data/tank_simulation/norm_long.csv').iloc[:, :3].reset_index(drop=True)

    # BeRfiPl data
    _, data, _, _, _ = preprocess_data('BeRfiPl_labels')
    df = pd.DataFrame(data)

    # Siemens data
    #df = pd.read_csv('preprocessed_data/SmA/id1_norm.csv').drop(columns=['CuStepNo ValueY']).reset_index(drop=True)
    #df_train, df_val = train_test_split(df, test_size=0.2)

    df_sc = pd.DataFrame(standardize_data(df, f'scaler_dataset_train.pkl'), columns=df.columns)

    data_module = DataModule(hparams=hparams, dataframe=df_sc)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=2)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode='min', patience=30)
    trainer = pl.Trainer(max_epochs=500, log_every_n_steps=1, logger=logger, accelerator='gpu', devices=1, callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, data_module)

if __name__ == '__main__':
    # complete code execution by use of the experiments method to compute several experiments after each other
    experiments_grid = make_grid()
    experiments_to_json(experiments_grid)
    experiments = load_experiments(modus='run')
    for experiment in range(len(experiments)):
        print('experiment no.', experiment)
        train(hparams=experiments[str(experiment)])
        print('completed experiment no ' + str(experiment + 1) + str(len(experiments)) + ' experiments')

