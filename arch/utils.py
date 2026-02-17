import numpy as np
import torch

def img_to_gal_axis(g1, g2, theta):
    '''
    Convert from (g1, g2) to (g_plus, g_cross) given position angle theta.
    '''
    # check if numpy array or torch tensor
    if isinstance(g1, torch.Tensor):
        g_plus = g1 * torch.cos(2*theta) - g2 * torch.sin(2*theta)
        g_cross = g1 * torch.sin(2*theta) + g2 * torch.cos(2*theta)
        return g_plus, g_cross
    else:
        g_plus = g1 * np.cos(2*theta) - g2 * np.sin(2*theta)
        g_cross = g1 * np.sin(2*theta) + g2 * np.cos(2*theta)
    return g_plus, g_cross

def gal_to_img_axis(g_plus, g_cross, theta):
    '''
    Convert from (g_plus, g_cross) to (g1, g2) given position angle theta.
    '''
    # check if numpy array or torch tensor
    if isinstance(g_plus, torch.Tensor):
        g1 = g_plus * torch.cos(2*theta) + g_cross * torch.sin(2*theta)
        g2 = -g_plus * torch.sin(2*theta) + g_cross * torch.cos(2*theta)
        return g1, g2
    else:
        g1 = g_plus * np.cos(2*theta) + g_cross * np.sin(2*theta)
        g2 = -g_plus * np.sin(2*theta) + g_cross * np.cos(2*theta)
    return g1, g2