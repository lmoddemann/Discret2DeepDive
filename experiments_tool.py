import json


def make_grid():
    """
    creates a grid of experiments from the parameter lists batch_size, ...

    :return: experiments_dict
"""
   
    # In our experiments, we varied the parameters of temperature and beta.
    # The results were tested with a temperature between 0.1 and 1 with incremental gradations of 0.1 
    # With beta, the influence was also examined in the range between 0.1 and 1 with steps of 0.1.

    temperature = [0.1, 0.3, 0.5, 0.7, 0.9]
    beta = [0.1, 0.3, 0.5, 0.7, 0.9]
    enc_out_dim = [1,2,4,8,16]
    dec_out_dim = [1,2,4,8,16]
    experiments_dict = {}

    l = 0
    
    # Training loops:
    training_loops = 10
    for i in range(training_loops):
            experiments_dict[l] = {
                "IN_DIM": 3, # 3 for tank, 39 for siemens, 16 for artificial, 104 for berfipl label
                "ENC_OUT_DIM": 8, 
                "DEC_OUT_DIM": 4, 
                "LATENT_DIM": 12,
                "CATEGORICAL_DIM": 12,
                "TEMPERATURE": .1, 
                "SOFTCLIP_MIN":0,
                "BATCH_SIZE": 512, 
                "BETA": .5, 
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