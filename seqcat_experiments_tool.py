import json


def make_grid():
    """
    creates a grid of experiments from the parameter lists batch_size, learning_rate, ...

    :return: experiments_dict
"""

    beta = [0.01, 0.1, 0.2, 0.5, 0.7, 0.9]
    temp = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    categorical_dim = [10, 20, 40]
    timesteps = [10, 20, 40, 80]
    hiddens = [8, 16, 32, 128, 256]
    enc_out = [4,8,16,32,64]

    experiments_dict = {}

    l = 0
    trainings = 10
    for i in range(trainings):
            experiments_dict[l] = {
                "NUMBER_TIMESTEPS" : 10,
                "SEQ_LEN" : 10,
                "VALID_SPLIT" : 0.2,
                "BETA" : .1, 
                "TEMP" : .7, 
                "BATCH_SIZE" : 512,
                "NUM_WORKERS" : 20,
                "CATEGORICAL_DIM" : 12, 
                "ENC_OUT_DIM": 64, 
                "IN_DIM" : 104, 
                "HIDDEN_DIM" : 32, 
                "RNN_MODELS" : True,
                "RNN_LAYERS" : 1,
                "Experiment_ID": l

                }
            l +=1
    return experiments_dict


def experiments_to_json(experiment_dict: dict):
    """
    writing the experiment_dict into a .json file for documentation

    :param experiment_dict:
    :return: .json file containing the dictionary of hyperparameters
    """
    experiments = experiment_dict
    with open("experiments_embed.json", "w") as json_file:
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
        with open("experiments_embed.json") as json_file:
            hparam = json.load(json_file)
    if "test" in modus:
        with open("testhparams.json") as json_file:
            hparam = json.load(json_file)
    return hparam


if __name__ == "__main__":
    experiments = make_grid()
    experiments_to_json(experiments)
    hparam = load_experiments(modus="run")
