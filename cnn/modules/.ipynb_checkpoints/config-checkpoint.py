import numpy as np

nGPUs = 2

data = \
{
    'size': 1000000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_train_low_g.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/train_data_low_g/train_database_5noise_all',
    'data_stem': 'training_'
}

test = \
{
    'size': 100000,
    'nimgs': 3,
    'nspec': 3,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/samples_test_low_g.csv',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/test_data_low_g/test_database_5noise_all',
    'data_stem': 'testing_'
}

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

train = \
{

    'epoch_number': 60,
    'initial_learning_rate': 0.001,
    'momentum': 0.9,
    
    'batch_size': 100,
    'feature_number': 8,
    
    'device': ['cuda:0', 'cuda:1'],
    'gpu_number': nGPUs,
    
    'save_model': True,
    'model_path': '/xdisk/timeifler/wxs0703/kl_nn/model/',
    'model_name': '5noise_all',
    
}
