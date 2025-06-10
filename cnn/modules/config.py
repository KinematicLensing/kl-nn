import numpy as np

data = \
{
    'size': 1000000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_train_low_g_1m.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/train_data/train_database_random_SNR',
    'data_stem': 'training_'
}

test = \
{
    'size': 100000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_test_low_g_1m.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/test_data/test_database_random_SNR',
    'data_stem': 'testing_'
}

par_ranges = \
{
    'g1': [-0.1, 0.1],
    'g2': [-0.1, 0.1],
    #'theta_int': [-np.pi, np.pi],
    'sini': [0, 1],
    'v0': [-30, 30],
    'vcirc': [60, 540],
    'rscale': [0.1, 10],
    #'hlr': [0.1, 5],
}

train = \
{

    'epoch_number': 60,
    'initial_learning_rate': 0.001,
    'momentum': 0.9,
    
    'batch_size': 100,
    'feature_number': 6,
    
    'save_model': True,
    'model_path': '/xdisk/timeifler/wxs0703/kl_nn/model/',
    'model_name': 'random_SNR',
    
}

cali = \
{
    'train_size': 200000,
    'test_size': 100000,
    
    'epoch_number': 60,
    'batch_size': 100,
    'feature_number': 7,
    
    'train_dir': '/xdisk/timeifler/wxs0703/kl_nn/train_data_200k/train_database_random_SNR',
    'test_dir': '/xdisk/timeifler/wxs0703/kl_nn/test_data/test_database_random_SNR',
}
