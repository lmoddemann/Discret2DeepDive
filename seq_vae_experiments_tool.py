import json


def make_grid():
    """
    creates a grid of experiments from the parameter lists batch_size, learning_rate, ...

    :return: experiments_dict
"""
    beta = [0.01, 0.1, 0.2, 0.5, 0.7, 1]
    latent_dim = [1, 2, 5]
    enc_out = [5, 10, 1]
    hidden = [1, 5, 10]
    experiments_dict = {}
    l = 0

    for k in enc_out:
        for m in hidden: 
                experiments_dict[l] = dict(
                                            valid_split=0.2,
                                            batch_size=256,
                                            num_workers=20,
                                            number_timesteps = 10, # must be same to seq_len
                                            enc_out_dim = k, 
                                            learning_rate=1e-3, 
                                            latent_dim=5, 
                                            input_size=3, #100 for berfipl labeled, 3 for tank, 39 for siemens
                                            seq_len=10, 
                                            beta=0.1, 
                                            hidden_size=m, 
                                            rnn_layers=1,
                                            rnn_models=True,
                )
                l +=1
    return experiments_dict


def experiments_to_json(experiment_dict: dict):
    """
    writing the experiment_dict into a .json file for documentation

    :param experiment_dict:
    :return: .json file containing the dictionary of hyperparameters
    """
    experiments = experiment_dict
    with open("../experiments_embed.json", "w") as json_file:
        json.dump(experiments, json_file, indent=4)
    print("experiments.json was created")
    return

def load_experiments(modus='run'):
    """
    reading the experiment .json file and returning a dictionary

    :param modus: (str) contains whether testing or training hyperparameters shall be used.
    :return: hparam <- dictionary with the content of the .json file
    """
    if "run" in modus:
        with open("../experiments_embed.json") as json_file:
            hparam = json.load(json_file)
    if "test" in modus:
        with open("testhparams.json") as json_file:
            hparam = json.load(json_file)
    return hparam


if __name__ == "__main__":
    experiments = make_grid()
    experiments_to_json(experiments)
    hparam = load_experiments(modus="run")
    print("funktioniert")
