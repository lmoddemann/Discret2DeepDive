import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import _pickle as pkl
import torch
import os


def standardize_data(data, name_pkl):
    # Standardize the data
    scaler = StandardScaler().fit(data)
    data = pd.DataFrame(scaler.transform(data), columns=data.columns, index=data.index)
    # save scaler
    path = Path("scaler")
    with open(Path(path, name_pkl), 'wb') as f:
        pkl.dump(scaler, f)
    return data

def standardize_seq(data, name_pkl):
    scaler = StandardScaler().fit(data)
    data = pd.DataFrame(scaler.transform(data))
    path = Path("scaler")
    with open(Path(path, name_pkl), 'wb') as f:
        pkl.dump(scaler, f)
    return data

def load_scaler(scaler_name):
    # Loading the saved scaler from the location "scaler" using the identifier "scaler_name"
    path = Path("scaler")
    file = Path(path, scaler_name)
    if not file.exists():
        raise ValueError("No scaler found at path:", file)
    with open(file, 'rb') as f:
        scaler = pkl.load(f)
    return scaler


def load_data(keyword):
    """
    Load data from CSV files in a specified directory based on the provided keyword.

    Args:
        - keyword (str): A keyword to identify the directory containing CSV files.

    Returns:
        - dataframes (list): A list of Pandas DataFrames loaded from CSV files.
    """

    if 'siemens' in keyword: 
        path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/preprocessed_data/SmA/id1_norm.csv')
        df = pd.read_csv(f'{path}').reset_index(drop=True)

    elif 'simu_tank' in keyword:
        path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/preprocessed_data/tank_simulation/norm.csv')
        df = pd.read_csv(f'{path}').iloc[1000:2000].reset_index(drop=True)


    else:
        # Expand the user's home directory and create the full path to the directory containing CSV files
        path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/preprocessed_data/{keyword}/')

        # List all file names in the specified directory that are files (not directories)
        file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and 'json' not in name]

        # Initialize an empty list to store DataFrames
        dataframes = []
        file_names = sorted(file_names, key=lambda x: int(x.split('n')[-1].split('.')[0]))

        # Loop through each file name and read the corresponding CSV file into a DataFrame
        for file_name in file_names:
            df = pd.read_csv(os.path.join(path, file_name), index_col=0)  # Use os.path.join to create the full file path
            dataframes.append(df)

        df = pd.concat(dataframes, axis=0).reset_index(drop=True)
    return df


def preprocess_data(file_path):
    """
    Preprocess the data loaded using the provided file path.

    Args:
        - file_path (str): The file path to the folder where the data is stored.

    Returns:
        - time (list): A list of the time sequences of the data.
        - train_list (list): A list of the sequences of the data.
        - state_list (list): A list of the state sequences of the data.
        - train_names (list): A list of the names of the data variables (except time and states).
        - state_names (list): A list of the names of the states.
    """
    # Load data from file
    data = load_data(file_path)

    # Extract time from the data
    time = torch.tensor(data['time'].values, dtype=torch.float)

    # Initialize lists
    train_list = []
    state_list = []
    train_columns = None
    state_columns = None
    num_nan_or_inf = 0

    dataframe=data
    # Get all the date except the time and state information and convert it to PyTorch tensor
    new_column_names = [col.replace('opening', 'alter') for col in dataframe.columns]
    dataframe.columns = new_column_names

    train_columns = ~dataframe.columns.str.contains('condition|open|time|Unnamed')
    train_columns = dataframe.loc[:, train_columns]
    train_values = torch.tensor(train_columns.values, dtype=torch.float)
    train_list.append(train_values)

    # Get all the state information from the dataset and convert it to PyTorch tensor
    state_columns = dataframe.columns[dataframe.columns.str.contains('open|condition')]
    state_columns = dataframe.loc[:, state_columns]
    signal_values = torch.tensor(state_columns.values, dtype=torch.float)
    state_list.append(signal_values)

    # Names of the data signals
    train_names = train_columns.columns.tolist()
    # Name of the state signals
    signal_names = state_columns.columns.tolist()

    return time, train_list[0], state_list[0], train_names, signal_names

# def preprocess_data_berfipl_anom(id):
#     """
#     Preprocess the data loaded using the provided file path.

#     Args:
#         - file_path (str): The file path to the folder where the data is stored.

#     Returns:
#         - time (list): A list of the time sequences of the data.
#         - train_list (list): A list of the sequences of the data.
#         - state_list (list): A list of the state sequences of the data.
#         - train_names (list): A list of the names of the data variables (except time and states).
#         - state_names (list): A list of the names of the states.
#     """
#     # Load data from file
#     data= pd.read_csv(f'preprocessed_data/BeRfiPl/{id}.csv').iloc[:, :].reset_index(drop=True)

#     # Initialize lists
#     train_list = []
#     state_list = []
#     train_columns = None
#     state_columns = None
#     num_nan_or_inf = 0

#     dataframe=data
#     # Get all the date except the time and state information and convert it to PyTorch tensor
#     new_column_names = [col.replace('opening', 'alter') for col in dataframe.columns]
#     dataframe.columns = new_column_names

#     train_columns = ~dataframe.columns.str.contains('condition|open|time|Unnamed')
#     train_columns = dataframe.loc[:, train_columns]
#     train_values = torch.tensor(train_columns.values, dtype=torch.float)
#     train_list.append(train_values)

#     # Get all the state information from the dataset and convert it to PyTorch tensor
#     state_columns = dataframe.columns[dataframe.columns.str.contains('open|condition')]
#     state_columns = dataframe.loc[:, state_columns]
#     signal_values = torch.tensor(state_columns.values, dtype=torch.float)
#     state_list.append(signal_values)

#     # Names of the data signals
#     train_names = train_columns.columns.tolist()
#     # Name of the state signals
#     signal_names = state_columns.columns.tolist()

#     return signal_names, train_list[0], state_list[0], train_names, signal_names

# def preprocess_data_siemens(file_path):
#     data = load_data(file_path)

#     time = torch.tensor(data.index)
#     train_list = torch.tensor(data.drop(columns=['CuStepNo ValueY']).to_numpy())
#     state_list = torch.tensor(data['CuStepNo ValueY'].reset_index(drop=True).to_numpy())

#     # Normalize the data
#     mean = torch.mean(train_list)
#     std = torch.std(train_list)
#     train_list = [(x - mean) / std for x in train_list]

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[0]

#     print('Data preprocessing done!')
#     return time, train_list, state_list, train_names, state_names

# def preprocess_data_siemens_anom(id):
#     # data = load_data(file_path)
#     data= pd.read_csv(f'preprocessed_data/SmA/{id}_anomaly.csv').iloc[:, :].reset_index(drop=True)

#     time = torch.tensor(data.index)
#     train_list = torch.tensor(data.drop(columns=['CuStepNo ValueY']).to_numpy())
#     state_list = torch.tensor(data['CuStepNo ValueY'].reset_index(drop=True).to_numpy())

#     # Normalize the data
#     mean = torch.mean(train_list)
#     std = torch.std(train_list)
#     train_list = [(x - mean) / std for x in train_list]

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[0]

#     print('Data preprocessing done!')
#     return time, train_list, state_list, train_names, state_names

# def preprocess_data_siemens_noscaler(file_path):
#     data = load_data(file_path)

#     time = torch.tensor(data.index)
#     train_list = torch.tensor(data.drop(columns=['CuStepNo ValueY']).to_numpy())
#     state_list = torch.tensor(data['CuStepNo ValueY'].reset_index(drop=True).to_numpy())

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[0]

#     print('Data preprocessing done!')
#     return time, train_list, state_list, train_names, state_names

# def preprocess_data_simu_tank(file_path):
#     data = load_data(file_path)

#     time = torch.tensor(data.index)
#     train_list = torch.tensor(data.iloc[:, :3].to_numpy())
#     state_list = data.iloc[:, 3].reset_index(drop=True)

#     # Normalize the data
#     mean = torch.mean(train_list)
#     std = torch.std(train_list)
#     train_list = [(x - mean) / std for x in train_list]

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[0]

#     print('Data preprocessing done!')
#     return time, train_list, state_list, train_names, state_names

# def preprocess_data_simu_tank_anom(id):
#     data = pd.read_csv(f'preprocessed_data/tank_simulation/{id}_faulty.csv').iloc[1000:4000, :].reset_index(drop=True)
#     time = torch.tensor(data.index)
#     train_list = torch.tensor(data.iloc[:, :3].to_numpy())
#     state_list = data.iloc[:, 3].reset_index(drop=True)

#     # Normalize the data
#     mean = torch.mean(train_list)
#     std = torch.std(train_list)
#     train_list = [(x - mean) / std for x in train_list]

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[0]

#     print('Data preprocessing done!')
#     return time, train_list, state_list, train_names, state_names

# def load_data_artificial(keyword):

#     path = os.path.expanduser(f'~/{str(Path.cwd().relative_to(Path.home()))}/preprocessed_data/{keyword}/')

#     # List all file names in the specified directory that are files (not directories)
#     file_names = [name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name)) and 'json' not in name]

#     # Initialize an empty list to store DataFrames
#     dataframes = []

#     # Loop through each file name and read the corresponding CSV file into a DataFrame
#     for file_name in file_names:
#         df = pd.read_csv(os.path.join(path, file_name))  # Use os.path.join to create the full file path
#         dataframes.append(df)

#     df = pd.concat(dataframes, axis=0).reset_index(drop=True).iloc[:50000]
#     return df


# def preprocess_data_artificial(file_path):

#     data = load_data_artificial(file_path)

#     # Extract time from the dataset
#     time = torch.tensor(data['Time'].values, dtype=torch.float)
#     # Get the values from the dataset except the times and states and convert them to PyTorch tensor
#     train_list = torch.tensor(data.values[:, 1:-1], dtype=torch.float)
#     # Get the state information from the dataset and convert it to PyTorch tensor
#     state_list = torch.tensor(data.values[:, -1], dtype=torch.float)

#     # Normalize the data
#     mean = torch.mean(train_list)
#     std = torch.std(train_list)
#     train_list = [(x - mean) / std for x in train_list]

#     # Names of the data signals
#     train_names = (data.columns.to_list())[1:-1]
#     # Name of the state signals
#     state_names = (data.columns.to_list())[-1]

#     print('Data preprocessing done!')

#     return time, train_list, state_list, train_names, state_names