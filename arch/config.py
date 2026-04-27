import numpy as np

sigma = 69.00924072265624 # spread of gaussian noise for a flux factor of 1, i.e. SNR=5 at rmag=23.4

# Training data info and locations
data = \
{
    'size': 1000000,
    'nimg': 1,
    'nspec': 5,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/train_tng_10k/',
    'data_stem': 'gal_',
}

# Validation data info and locations
test = \
{
    'size': 100000,
    'nimg': 1,
    'nspec': 5,
    'data_dir': '/ocean/projects/phy250048p/shared/datasets/test_tng_10k/',
    'data_stem': 'gal_'
}

# Which parameters should the CNN predict and what are their prior ranges?
par_ranges = \
{
    'g1': [-0.1, 0.1],
    'g2': [-0.1, 0.1],
    # 'theta_int': [0, np.pi],
    # #'sin_theta': [-1, 1],
    # #'cos_theta': [-1, 1],
    # 'sini': [0, 1],
    # 'v0': [-30, 30],
    'vcirc': [60, 540],
    # 'rscale': [0.1, 10],
    # 'hlr': [0.1, 1],
}

# CNN model training metaparameters
train = \
{

    'mode': 2,  # 0: point estimate; 1: density estimate via normalizing flow; 2: density estimate with TF prior
    'epoch_number': 100,
    'initial_learning_rate': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-5,
    
    'batch_size': 100,
    'feature_number': 3,
    
    'save_model': True,
    'model_path': '/ocean/projects/phy250048p/shared/models/',
    'model_name': 'CNN-CNN-flow_tf_tng',
    'transform_to_gal': False,

    'use_pretrain': True,
    'pretrained_name': 'CNN-CNN-flow_1m_tf',
    'pretrain_from': 149
    
}

flow = \
{
    'num_layers': 5,
    'mlp': [1, 128, 64, 2],
}