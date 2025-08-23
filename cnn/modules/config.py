import numpy as np

# Training data info and locations
data = \
{
    'size': 5000000,
    'nimgs': 3,
    'nspec': 3,
    
    'data_dir': '/data/wxs0703/kl-nn/databases/train_database_1m',
    'data_stem': 'training_'
}

# Validation data info and locations
test = \
{
    'size': 500000,
    'nimgs': 3,
    'nspec': 3,
    
    'data_dir': '/data/wxs0703/kl-nn/databases/test_database_1m',
    'data_stem': 'testing_'
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

    'epoch_number': 100,
    'initial_learning_rate': 0.1,
    'momentum': 0.9,
    
    'batch_size': 100,
    'feature_number': 8,
    
    'save_model': True,
    'model_path': '/data/wxs0703/kl-nn/models/',
    'model_name': 'test_1m_noise_nonorm',
    'use_pretrain': False,
    'pretrained_name': 'randSNR_noiseless',
    'pretrain_from': 46
    
}

# Calibration network training metaparameters
cali = \
{
    'epoch_number': 30,
    'learning_rate': 0.0001,
    'batch_size': 100,
    'feature_number': 8,
    
    'model_path': '/data/wxs0703/kl-nn/models/',
    'model_name': 'test_1m_noise_nonorm',
    'data_dir': '/data/wxs0703/kl-nn/databases/cali_database_5m',
    'res_dir': '/data/wxs0703/kl-nn/cali/'
}
