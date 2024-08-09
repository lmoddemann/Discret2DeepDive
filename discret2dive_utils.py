import pandas as pd 
import numpy as np
import yaml 
import torch
import json
import os
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns.fpgrowth import fpgrowth
from mlxtend.frequent_patterns import association_rules

from utils import standardize_data
from seqcat_catvae import seqcat_vae as Seq_CatVAE
from seq_vae import seq_vae as Seq_VAE
from seq_datamodule import Dataset as seq_dataset
from check_truth import check_true_cats


def load_model():
    """ Loading the trained CatVAE from the location "VAE_training_hparams" """

    MODEL_VERSION = f'VAE_training_hparams/tank_discret2deepdive'
    ckpt_file_name = os.listdir(f'./{MODEL_VERSION}/checkpoints/')[-1]
    ckpt_file_path = f'./{MODEL_VERSION}/checkpoints/{ckpt_file_name}'
    with open(f'./{MODEL_VERSION}/hparams.yaml') as f:
        hparam_disc = yaml.safe_load(f)
    model_disc = Seq_CatVAE.load_from_checkpoint(ckpt_file_path, hparams=hparam_disc["hparams"])

    MODEL_VERSION_SEQ = f'VAE_training_hparams/tank/seq_vae'
    ckpt_file_name_seq = os.listdir(f'./{MODEL_VERSION_SEQ}/checkpoints/')[-1]
    ckpt_file_path_seq = f'./{MODEL_VERSION_SEQ}/checkpoints/{ckpt_file_name_seq}'
    with open(f'./{MODEL_VERSION_SEQ}/hparams.yaml') as f:
        hparam_res = yaml.safe_load(f)
    model_res = Seq_VAE.load_from_checkpoint(ckpt_file_path_seq, hparams=hparam_res)

    return model_disc, hparam_disc, model_res, hparam_res

def generate_discretizatons_norm(model_disc, hparam_disc, model_res, hparam_res):
    """ Generate discretizations of the differnt datasets (nominal/anomalous system behavior). 
    In doing so, we discard the first 1000 samples in order to concentrate on the steady state."""

    df = pd.read_csv('preprocessed_data/tank_simulation/norm_long.csv').iloc[1000:, :3].reset_index(drop=True)
    df_scale = standardize_data(df, 'diagnosis_scaler.pkl')
    dataset_seq = seq_dataset(dataframe=pd.DataFrame(df_scale), **hparam_res)
    
    # Generate likelihoods and discretizations for data windows of equal length for seq_catvae and seq_vae
    total_residuals = []
    total_disc = []
    for window in dataset_seq:
        # generate the residuals by means of likelihood computation
        _, loss_dict = model_res(window.unsqueeze(0).to('cuda'))
        total_residuals.extend(list(loss_dict.items())[-1][1].flatten().detach().cpu().numpy())
        # compute discretizations in terms of extracting categories with the RNN-based CatVAE
        _, _, _, _, _, z_states = model_disc.get_states(window.unsqueeze(0).to('cuda'))
        total_disc.append(z_states.detach().cpu().numpy().astype(int))

    df_states = pd.DataFrame(np.vstack(total_disc))
    cats = pd.DataFrame(df_states.idxmax(axis=1))
    unique_disc = cats[cats.columns[0]].unique()

    all_residuals = np.array(total_residuals)
    threshold_res = all_residuals.min() - 10

    # Discretization of the likelihood
    disc_likelihood = pd.DataFrame(np.where(all_residuals < threshold_res, '1', '0'), columns=["likelihood"])
    # convert into str since simpy solver can not handle integers
    disc_states = df_states.astype(str).agg(''.join, axis=1)
    states = np.array(pd.concat([disc_states, disc_likelihood], axis=1))
    disc_states = df_states.astype(str).agg(''.join, axis=1)
    unique_disc = list(disc_states.unique())
    disc_char = disc_states.str.replace('0', 'a').str.replace('1', 'b')
    states = np.array(pd.concat([disc_states, disc_likelihood], axis=1))
    resulting_states = pd.concat([disc_char, disc_likelihood], axis=1).rename(columns={0:'cats'})

    return states, resulting_states, unique_disc, df, disc_likelihood, all_residuals, threshold_res


def assosciation_rule_mining(states, unique_disc, min_support):
    """ Association Rule mining from discretized "states" with FPGrowth."""
    # Transform the dataset into an array format suitable 
    states_encoder = TransactionEncoder()
    states_encoder.fit(states)
    encoded_states = states_encoder.transform(states)
    df_encoded_states = pd.DataFrame(encoded_states, columns=states_encoder.columns_)
    # Learning of frequent itemsets
    frequent_itemsets = fpgrowth(df_encoded_states, min_support=min_support, use_colnames=True)
    # Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="support", min_threshold=min_support)
    # Filter out the rules by its implicants (unique discretizations)
    true_rules_bool = pd.DataFrame([rules[rules.columns[0]] == frozenset({f'{comp}'}) for comp in unique_disc]).T.any(axis=1)
    true_rules = rules[true_rules_bool.values]
    return true_rules, rules


def create_dict(true_rules):
    """ create dictionary of categories to its discretized likelihoods"""
    states_disc = list(true_rules[true_rules.columns[0]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    res_disc = list(true_rules[true_rules.columns[1]].apply(lambda x: ', '.join(list(x))).astype("unicode"))
    df_disc = pd.DataFrame({"states_disc":states_disc, "res_disc":res_disc})
    states_dict = df_disc.groupby('states_disc')['res_disc'].apply(list).to_dict()
    for state in list(states_dict.keys()): 
        dict_states_char = {key.replace('0', 'a').replace('1', 'b'): value for key, value in states_dict.items()}
    return dict_states_char


def save_files(true_rules, dict_states_char, dummy, dict_path, rule_path):
    """ saving rules at path "diagnosis" """
    json.dump(dict_states_char, open(f"diagnosis/{dict_path}.txt",'w'))
    with open(f'diagnosis/{rule_path}.txt', 'w') as f:
        [f.writelines([f"dummy IMPLIES {key}\n" for (key, value) in dict_states_char.items()])]
    return print("rules generated and saved")


def check_truth(resulting_states, dict_states_char):
    """ check resulting states, if included in dictionary or whether category not included within rule base"""
    true_vals = []
    not_found = []
    for result_states0, result_states1 in zip(resulting_states[resulting_states.columns[0]], resulting_states[resulting_states.columns[1]]):
        if dict_states_char.get(result_states0) is None:
            not_found.append(result_states0)
            res = False 
            true_vals.append(res)
            continue
        else:
            res = result_states1 in dict_states_char.get(result_states0)
            true_vals.append(res)
    resulting_states = pd.concat([resulting_states, pd.DataFrame(true_vals, index=resulting_states.index, columns=['results'])], axis=1)
    return resulting_states

