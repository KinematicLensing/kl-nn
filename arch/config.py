from os.path import dirname, join

import numpy as np

sigma = 69.00924072265624 # spread of gaussian noise for a flux factor of 1, i.e. SNR=5 at rmag=23.4

rmag_snr_source_path = '/ocean/projects/phy250048p/shared/temp/rmag_snr_pv.npz'
rmag_snr_fit_path = join(dirname(__file__), 'rmag_snr_fit.npz')

# Training data info and locations
data = \
{
    'size': 1000000,
    'nimg': 1,
    'nspec': 5,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/train_1m_low_hlr/',
    'data_stem': 'gal_',
}

# Validation data info and locations
test = \
{
    'size': 100000,
    'nimg': 1,
    'nspec': 5,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/test_1m_low_hlr/',
    'data_stem': 'gal_'
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
    'hlr': [0.1, 1],
}

# CNN model training metaparameters
train = \
{

    'mode': 2,  # 0: point estimate; 1: density estimate via normalizing flow; 2: density estimate with TF prior
    'epoch_number': 200,
    'initial_learning_rate': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    
    'batch_size': 100,
    'feature_number': 8,
    'feature_names': ['g1', 'g2', 'theta_int', 'sini', 'v0', 'vcirc', 'rscale', 'hlr'],
    
    'save_model': True,
    'model_path': '/ocean/projects/phy250048p/shared/models/',
    'model_name': 'CNN-CNN-flow_all_params_unif_snr_fib_pos',
    'transform_to_gal': False,

    'use_pretrain': False,
    'pretrained_name': 'CNN-CNN-flow_all_params',
    'pretrain_from': 99
    
}

flow = \
{
    'num_layers': 12,
    'mlp': [1, 128, 64, 2],
}