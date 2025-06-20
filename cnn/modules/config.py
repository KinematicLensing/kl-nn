import numpy as np

# Training data info and locations
data = \
{
    'size': 5000000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_train_5m.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/databases/train_database_5m',
    'data_stem': 'training_'
}

# Validation data info and locations
test = \
{
    'size': 500000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_test_5m.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/databases/test_database_5m',
    'data_stem': 'testing_'
}

# Which parameters should the CNN predict and what are their prior ranges?
par_ranges = \
{
    'g1': [-0.1, 0.1],
    'g2': [-0.1, 0.1],
    'theta_int': [-np.pi, np.pi],
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
    'initial_learning_rate': 0.01,
    'momentum': 0.9,
    
    'batch_size': 100,
    'feature_number': 8,
    
    'save_model': True,
    'model_path': '/xdisk/timeifler/wxs0703/kl_nn/model/',
    'model_name': 'randSNR_5m',
    
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
