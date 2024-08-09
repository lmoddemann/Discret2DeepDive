import torch
import pandas as pd
import numpy as np

def check_true_cats(model_res, hparam_res, data, tolerance):
    
    # Generate discretizations/categories
    pzx_logits, pzx, mu, sigma, pxz, z = model_res.get_states(torch.tensor(data).to(device='cuda'))  
    df_states = pd.DataFrame(torch.zeros(z.shape).to(device='cuda').scatter(1, torch.argmax(pzx_logits, dim=1).unsqueeze(1), 1).cpu().detach().numpy(), index=pd.DataFrame(data).index).astype(int)
    disc_states = df_states.astype(str).agg(''.join, axis=1)
    unique_disc = list(disc_states.unique())
    df_disc = pd.DataFrame([list(disc) for disc in unique_disc]).astype(np.float32)

    # check mean values of discretizations and decide on a tolerance, whether the category is relevant or not
    mu_arr = pd.DataFrame()
    idx_arr = []
    for i in range(len(unique_disc)):
        data = df_disc.iloc[i]
        pxz, mu = model_res.generate(torch.tensor(data).to(device='cuda').view(1,hparam_res["hparams"]["CATEGORICAL_DIM"]))
        mu_arr = pd.concat([mu_arr, pd.DataFrame(pxz.mean.cpu().detach().numpy()[0]).T])
        idx_arr.append(data.argmax())
    df_mu = mu_arr.set_index(pd.Index(idx_arr))
    bool_arr = pd.DataFrame()
    idx_arr = []
    for i in range(len(df_mu)):
        for j in range(i+1,len(df_mu)):
            ax = pd.DataFrame(np.isclose(df_mu.iloc[i].to_numpy(), df_mu.iloc[j].to_numpy(), atol=tolerance)).T
            bool_arr = pd.concat([bool_arr, ax])
            idx_arr.append(f'{df_mu.index[i]}-{df_mu.index[j]}')
    truth_check = bool_arr.set_index(pd.Index(idx_arr))
    ret_val = truth_check.all(axis=1)
    return_val = ret_val.loc[ret_val==True].index
    dict_cats = {}
    if len(return_val) != 0:
        
        for key in df_disc.idxmax(axis=1):
            if int(return_val.values[0].split('-')[1]) == key: 
                dict_cats[f"{key}"] = int(return_val.values[0].split('-')[0])
    else: 
        dict_cats = {}
    return dict_cats
