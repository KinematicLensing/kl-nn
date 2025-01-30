nGPUs = 2

data = \
{
    'size': 10000,
    'nspec': 5,
    'img_index': 11,
    
    'pars_dir': '/xdisk/timeifler/wxs0703/kl_nn/samples/',
    'data_dir': '/xdisk/timeifler/wxs0703/kl_nn/fits/',
    'data_stem': 'training_'
}

train = \
{

    'epoch_number': 20,
    'initial_learning_rate': 0.01,
    'momentum': 0.9,
    
    'batch_size': 100,
    'validation_split': 0.1,
    'feature_number': 8,
    
    'device': ['cuda:0', 'cuda:1'],
    'gpu_number': nGPUs,
    
    'save_model': True,
    'model_path': '/xdisk/timeifler/wxs0703/kl_nn/model/',
    'model_name': 'test_model',
    
}

simulation = \
{

    'pixel_size': 0.074,
    'galaxy_stamp_size': 128,
    'psf_stamp_size': 48,
    
    'read_noise': 5.0,
    'sky_background': 31.8,
    'dark_noise': 2.6,
    'bias_level': 500,
    'gain': 1.1,
    
}
