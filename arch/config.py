import numpy as np

# Training data info and locations
data = \
{
    'size': 10000,
    'nimg': 1,
    'nspec': 3,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/small/',
    'data_stem': 'gal_'
}

# Validation data info and locations
test = \
{
    'size': 10000,
    'nimg': 1,
    'nspec': 3,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/small/',
    'data_stem': 'gal_'
}

# Which parameters should the CNN predict and what are their prior ranges?
par_ranges = \
{
    'g1': [-0.1, 0.1],
    'g2': [-0.1, 0.1],
    'theta_int': [0, np.pi],
    #'sin_theta': [-1, 1],
    #'cos_theta': [-1, 1],
    'sini': [0, 1],
    'v0': [-30, 30],
    'vcirc': [60, 540],
    'rscale': [0.1, 10],
    'hlr': [0.1, 5],
}

# CNN model training metaparameters
train = \
{

    'mode': 1,
    'epoch_number': 200,
    'initial_learning_rate': 1,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    
    'batch_size': 100,
    'feature_number': 2,
    
    'save_model': True,
    'model_path': '/ocean/projects/phy250048p/shared/models/',
    'model_name': 'small_test',
    'use_pretrain': False,
    'pretrained_name': 'randSNR_noiseless',
    'pretrain_from': 46
    
}

flow = \
{
    'num_layers': 32,
    'mlp': [1, 64, 64, 2],
}

# Calibration network training metaparameters
cali = \
{
    'epoch_number': 30,
    'learning_rate': 0.0001,
    'batch_size': 100,
    'feature_number': 8,
    'n_cases': 5000,
    'n_realizations': 1000,
    
    'model_path': '/data/wxs0703/kl-nn/models/',
    'model_name': 'cali_1m_noise_nonorm',
    'data_dir': '/data/wxs0703/kl-nn/databases/cali_database_5m',
    'res_dir': '/data/wxs0703/kl-nn/cali/'
}
