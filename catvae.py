import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from datamodule import DataModule
from experiments_tool import *


class CategoricalVAE(pl.LightningModule):
    def __init__(self,
                 hparams:dict,
                 **kwargs) -> None:

        # parameters from hparams dictionary
        self.in_dim = hparams["IN_DIM"]
        self.enc_out_dim = hparams["ENC_OUT_DIM"]
        self.dec_out_dim = hparams["DEC_OUT_DIM"]
        self.categorical_dim = hparams["CATEGORICAL_DIM"]
        self.temp = hparams["TEMPERATURE"]
        self.beta = hparams["BETA"]
        self.model = 'catvae'
        super(CategoricalVAE, self).__init__()
        self.save_hyperparameters()

        # Build Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.enc_out_dim))
        self.fc_z_cat = nn.Linear(self.enc_out_dim, self.categorical_dim)

        # Build Decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.categorical_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.dec_out_dim))
        self.fc_mu_x = nn.Linear(self.dec_out_dim, self.in_dim)
        self.fc_logvar_x = nn.Linear(self.dec_out_dim, self.in_dim)

        # Categorical prior
        self.pz = torch.distributions.OneHotCategorical(
            1. / self.categorical_dim * torch.ones(1, self.categorical_dim, device='cuda'))

        self.log_sigma = torch.nn.Parameter(torch.full((1,), 0.)[0], requires_grad=True )

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder
        :return z_out: (Tensor) Latent code
        """
        result = self.encoder(input)
        z = self.fc_z_cat(torch.flatten(result, start_dim=1))
        z_out = z.view(-1, self.categorical_dim)
        return z_out

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Computes parameters for pxz from sampels of pzx
        :param z: (Tensor) 
        :return: mu (Tensor) 
        :return: sigma (Tensor)
        """
        result = self.decoder(z)
        mu = self.fc_mu_x(result)
        logvar = self.fc_logvar_x(result)
        sigma = torch.cat(
                        [torch.diag(torch.exp(logvar[i, :])) for i in range(z.shape[0])]
                       ).view(-1, self.in_dim, self.in_dim)
        return mu, sigma

    def sample_gumble(self, logits: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
        """
        Gumbel-softmax trick to sample from Categorical Distribution
        :param logits: (Tensor) Latent Codes 
        :return: (Tensor)
        """
        # Sample from Gumbel
        u = torch.rand_like(logits)
        g = - torch.log(- torch.log(u + eps) + eps)
        s = F.softmax((logits + g) / self.temp, dim=-1)
        return s

    def shared_eval(self, x: torch.Tensor):
        """
        shared computation of all steps/methods in CatVAE
        """
        # first compute parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categorical dist. object for use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx 
        z = self.sample_gumble(logits=pzx_logits)
        # decode into mu and sigma
        mu, sigma = self.decode(z)
        # construct multivariate distribution object for pxz
        pxz = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z
    
    def get_states(self, x: torch.Tensor):
        """
        computation of discretized states
        :param x: (Tensor)
        :return:
        """
        # first compute parameters of categorical dist. pzx
        pzx_logits = self.encode(x)
        # create one hot categorical dist. object for later use in loss func
        pzx = torch.distributions.OneHotCategorical(logits=pzx_logits)
        # sample from pzx (one hot categorical)
        z = self.sample_gumble(logits=pzx_logits)
        # compute states by using the argmax of logits
        z_states = torch.zeros(z.shape).to(device='cuda').scatter(1, torch.argmax(pzx_logits, dim=1).unsqueeze(1), 1)
        # decode into mu and sigma
        mu, sigma = self.decode(z_states)
        # construct multivariate distribution object for pxz
        pxz = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=sigma)
        return pzx_logits, pzx, mu, sigma, pxz, z

    def training_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct = self.loss_function(x=x, pzx=pzx, pxz=pxz)
        self.log('Loss', loss_dct['Loss'])
        self.log('recon_loss', loss_dct['recon_loss'])
        self.log('KLD_cat', loss_dct['KLD_cat'])
        self.log('train_loss', loss_dct['Loss'])
        return loss_dct['Loss']

    def validation_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct_val = self.loss_function(x=x, pzx=pzx, pxz=pxz)
        self.log('Loss', loss_dct_val['Loss'])
        self.log('recon_loss', loss_dct_val['recon_loss'])
        self.log('KLD_cat', loss_dct_val['KLD_cat'])
        self.log('val_loss', loss_dct_val['Loss'])
        return loss_dct_val['Loss']

    def test_step(self, x, batch_idx):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        loss_dct_test = self.loss_function(x=x, pzx=pzx, pxz=pxz)
        self.log('Loss', loss_dct_test['Loss_test'])
        self.log('recon_loss', loss_dct_test['recon_loss'])
        self.log('KLD_cat', loss_dct_test['KLD_cat'])
        return loss_dct_test['Loss']

    def forward(self, x: torch.Tensor, **kwargs):
        return self.shared_eval(x)

    def loss_function(self, x: torch.Tensor,
                    pzx: torch.distributions.OneHotCategorical, 
                    pxz: torch.Tensor) -> dict:
    
        likelihood = pxz.log_prob(x)
        recon_loss = torch.mean(likelihood)
        # compute kl divergence for categorical dist
        kl_categorical = torch.distributions.kl.kl_divergence(pzx, self.pz)
        kl_categorical_batch = torch.mean(kl_categorical)
        loss = -recon_loss + self.beta*kl_categorical_batch
        return {'Loss': loss, 'recon_loss': recon_loss, 'KLD_cat': kl_categorical_batch}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
    def function_likelihood(self, x:torch.Tensor):
        pzx_logits, pzx, mu, sigma, pxz, z = self.shared_eval(x)
        likelihood = pxz.log_prob(x)
        return likelihood


def train(hparam:dict):
    import random
    random_seed = random.randint(0,10000)
    model_name = 'catvae_tank_cat12_v31'
    logger = TensorBoardLogger('lightning_logs', name=model_name, default_hp_metric=False)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    model = CategoricalVAE(hparams=hparam)
    # Decide on the different datasets by entering either 
    # 'Tank_normal', 'SWaT_norm', 'BeRfiPl_ds1n' or 'SmA_normal' or 'Artificial'
    #data_module = DataModule(hparam_batch=hparam['BATCH_SIZE'], dataset_name='BeRfiPl_ds1n_label')
    #data_module = DataModule(hparam_batch=hparam['BATCH_SIZE'], dataset_name='Artificial')
    #data_module = DataModule(hparam_batch=hparam['BATCH_SIZE'], dataset_name='siemens_discrompa')
    data_module = DataModule(hparam_batch=hparam['BATCH_SIZE'], dataset_name='Tank_normal')
    
    checkpoint_callback = ModelCheckpoint( monitor='val_loss' ,save_top_k=2)
    early_stop_callback = EarlyStopping(monitor="val_loss", mode='min', patience=30)
    trainer = pl.Trainer(max_epochs=400, log_every_n_steps=10, logger=logger, accelerator='gpu', devices=1, callbacks=[early_stop_callback, checkpoint_callback])
    trainer.fit(model, data_module)
    
if __name__ == '__main__':
    # complete code execution by use of the experiments method to compute several experiments after each other
    experiments_grid = make_grid()
    experiments_to_json(experiments_grid)
    experiments = load_experiments(modus='run')
    for experiment in range(len(experiments)):
        print('experiment no.', experiment)
        train(hparam=experiments[str(experiment)])
        print('completed experiment no ' + str(experiment + 1) + str(len(experiments)) + ' experiments')
