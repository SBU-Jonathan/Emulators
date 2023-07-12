# NN COLA Emulator
import os
import numpy as np
import keras
import train_utils as utils

path_to_emulator =  os.path.dirname(__file__)
cola_redshifts = np.loadtxt(f"{path_to_emulator}/data/cola_zs.txt")
cola_ks_default = np.loadtxt(f"{path_to_emulator}/data/cola_ks_default.txt")
cola_ks_high = None

def load_lcdm_def_models():
    """
    Description: returns an array of NN models for default-precision LCDM for each COLA redshift z.
    """
    nn_models = []
    for i, z in enumerate(cola_redshifts):
        nn_models.append(keras.models.load_model(f"{path_to_emulator}/models/LCDM/NN_Z{z:.3f}"))
    return nn_models

def load_wcdm_models():
    """
    Description: returns an array of NN models for default-precision wCDM for each COLA redshift z.
    """
    nn_models = []
    for i, z in enumerate(cola_redshifts):
        nn_models.append(keras.models.load_model(f"{path_to_emulator}/models/wCDM/NN_Z{z:.3f}"))
    return nn_models

def load_pcas():
    """
    Description: returns an array of PC basis and averages to inverse transform the NN outputs.
    For reference, the inverse transform of a set of components {PC_i} is:
        f = avg + \sum_i PC_i * basis_i
    """
    avgs = np.zeros((len(cola_redshifts), len(cola_ks_default)))
    pcs = np.zeros((len(cola_redshifts), 11, len(cola_ks_default)))
    for i, z in enumerate(cola_redshifts):
        if z < 2:
            avgs[i] = np.loadtxt(f"{path_to_emulator}/data/averages/avg_{i}.txt")
            pcs[i] = np.loadtxt(f"{path_to_emulator}/data/pc_basis/pcs_{i}.txt")
        else:
            avgs[i,:256] = np.loadtxt(f"{path_to_emulator}/data/averages/avg_{i}.txt")
            pcs[i,:,:256] = np.loadtxt(f"{path_to_emulator}/data/pc_basis/pcs_{i}.txt")
    return pcs, avgs
    
def load_normalization_factors():
    """
    Description: returns an array of normalization factors to rescale the Qs.
    """
    mins = np.loadtxt(f"{path_to_emulator}/data/mins.txt")
    maxs = np.loadtxt(f"{path_to_emulator}/data/maxs.txt")
    return mins, maxs

def inverse_transform(components, z_index):
    """
    Description: inverse transform PCA components into normalized Qs.
    """
    global pcs, avgs
    result = avgs[z_index]
    for i in range(11):
        result += components[i] * pcs[z_index, i]
    return result

def get_boost(cosmo_params, ks, zs):
    """
    Description: returns an array of boosts at given ks and zs for the cosmology defined in cosmo_params
    Inputs:
        - `cosmo_params`: a dictionary of cosmological parameters. The keys to be defined are the same as EE2: `Omm`, `Omb`, `ns`, `As`, `h`, `w`.
        - `ks`: array of scales to return
        - `zs`: array of redshifts to return
    """
    global maxs, mins, nn_models
    As = cosmo_params['As']
    Omega_m = cosmo_params['Omm']
    Omega_b = cosmo_params['Omb']
    ns = cosmo_params['ns']
    h = cosmo_params['h']
    norm_params = utils.normalize_params([Omega_m, Omega_b, ns, As, h])
    
    expqs = np.zeros((len(cola_redshifts), len(cola_ks_default)))
    
    for i, z in enumerate(cola_redshifts):
        principal_components = nn_models[i](np.array([norm_params]))[0]
        norm_qs = inverse_transform(principal_components, i)
        qs = norm_qs * (maxs[i] - mins[i]) + mins[i]
        expqs[i] = np.exp(qs)

    return expqs

mins, maxs = load_normalization_factors()
pcs, avgs = load_pcas()
nn_models = load_lcdm_def_models()