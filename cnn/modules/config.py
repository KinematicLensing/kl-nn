import numpy as np

# Training data info and locations
data = \
{
    'size': 5000000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_train_5m.csv',
    'data_dir': '/data/wxs0703/kl-nn/databases/train_database_box_noiseless',
    'data_stem': 'training_'
}

# Validation data info and locations
test = \
{
    'size': 500000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_test_5m.csv',
    'data_dir': '/data/wxs0703/kl-nn/databases/test_database_box_noiseless',
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

    'epoch_number': 30,
    'initial_learning_rate': 0.01,
    'momentum': 0.9,
    
    'batch_size': 200,
    'feature_number': 8,
    
    'save_model': True,
    'model_path': '/data/wxs0703/kl-nn/models/',
    'model_name': 'randSNR_box_noiseless',
    
}

# Calibration network training metaparameters
cali = \
{
    'train_size': 200000,
    'valid_size': 100000,
    
    'epoch_number': 60,
    'learning_rate': 0.0001,
    'batch_size': 100,
    'feature_number': 7,
    
    'train_dir': '/xdisk/timeifler/wxs0703/kl_nn/train_data_200k/train_database_random_SNR',
    'valid_dir': '/xdisk/timeifler/wxs0703/kl_nn/test_data/test_database_random_SNR',
}
