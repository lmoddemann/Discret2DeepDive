from discret2dive_utils import load_model, generate_discretizatons_norm, assosciation_rule_mining, create_dict, save_files, check_truth
from preprocessing_rules import getHealthStates
from diagnoser import Diag_solver
import json 
from translate import parse_file
import pandas as pd

def check_normal(min_support, dict_path, rule_path):
    """ Load the learned model of the CatVAE and generate discretizations and their according likelihoods based on the nominal system behavior. 
      Based on the discretizations, frequent itemsets can be found and formulated as rules with a dummy as non-observable components.
      """
    # load CatVAE model and its hyperparameter
    model_disc, hparam_disc, model_res, hparam_res = load_model()
    states, resulting_states, unique_disc, data_label, disc_likelihood, likelihood, threshold_res = generate_discretizatons_norm(model_disc=model_disc, 
                                                                                                        hparam_disc=hparam_disc,
                                                                                                        model_res=model_res,
                                                                                                        hparam_res=hparam_res)
    # calculation of the minimum support based on https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
    min_supp = (resulting_states[resulting_states.columns[0]].value_counts().min()/len(resulting_states))*min_support
    true_rules, rules = assosciation_rule_mining(states, unique_disc, min_supp)
    # create dictionary of categories to discretized likelihoods
    dict_states_char = create_dict(true_rules=true_rules)
    # dummy for non-observable components
    dummy_predicates = 'dummy'
    # saving rules at path "diagnosis"
    save_files(true_rules=true_rules, dict_states_char=dict_states_char, dummy=dummy_predicates, dict_path=dict_path, rule_path=rule_path)
    # check resulting states, if included in dictionary or whether category not included within rule base
    anomaly_df1 = pd.concat([data_label.reset_index(drop=True).shift(-hparam_res['seq_len']), resulting_states], axis =1)
    anomaly_df = anomaly_df1.iloc[:-hparam_res['seq_len']].reset_index(drop=True)
    return anomaly_df, likelihood, threshold_res
