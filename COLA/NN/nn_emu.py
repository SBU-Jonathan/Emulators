# NN COLA Emulator
import os
import numpy as np
import keras

path_to_emulator =  os.path.dirname(__file__)
cola_redshifts = np.loadtxt(f"{path_to_emulator}/data/cola_zs.txt")
cola_ks_default = np.loadtxt(f"{path_to_emulator}/data/cola_ks_default.txt")
cola_ks_high = None

def normalize_params(params):
    '''
    Takes a params = (h, Omegam, Omegab, As10to9, ns, wde) array and returns normalized parameters.
    Normalization is given by normalized_param = (param - param_min)(param_max - param_min).
    '''
    if len(params) == 6:
        Omegam, Omegab, ns, As10to9, h, wde = params
        normalized_params = [
            (h-lims['h'][0])/(lims['h'][1] - lims['h'][0]),
            (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
            (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
            (As10to9-lims['As'][0])/(lims['As'][1] - lims['As'][0]),
            (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
            (wde-lims['w'][0])/(lims['w'][1] - lims['w'][0])
        ]
    if len(params) == 5:
        Omegam, Omegab, ns, As10to9, h = params
        normalized_params = [
            (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
            (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
            (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
            (As10to9-lims['As'][0])/(lims['As'][1] - lims['As'][0]), 
            (h-lims['h'][0])/(lims['h'][1] - lims['h'][0])
        ]

    normalized_params = np.array(normalized_params)
    return normalized_params

def load_lcdm_def_models():
    """
    Returns an array of NN models for default-precision LCDM for each COLA redshift z.
    """
    nn_models = []
    for i, z in enumerate(cola_redshifts):
        nn_models.append(keras.models.load_model(f"{path_to_emulator}/models/LCDM/NN_Z{z:.3f}"))
    return nn_models

def load_wcdm_models():
    """
    Returns an array of NN models for default-precision wCDM for each COLA redshift z.
    """
    nn_models = []
    for i, z in enumerate(cola_redshifts):
        nn_models.append(keras.models.load_model(f"{path_to_emulator}/models/wCDM/NN_Z{z:.3f}"))
    return nn_models

def load_pcas():
    """
    Returns an array of PC basis and averages to inverse transform the NN outputs.
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
    Returns an array of normalization factors to rescale the Qs.
    """
    mins = np.loadtxt(f"{path_to_emulator}/data/mins.txt")
    maxs = np.loadtxt(f"{path_to_emulator}/data/maxs.txt")
    return mins, maxs

def inverse_transform(components, z_index):
    """
    Inverse transform PCA components into normalized Qs.
    """
    global pcs, avgs
    result = avgs[z_index]
    for i in range(11):
        result += components[i] * pcs[z_index, i]
    return result

def get_boost(cosmo_params, ks, zs):
    """
    Returns an array of boosts at given ks and zs for the cosmology defined in cosmo_params
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
    norm_params = normalize_params([Omega_m, Omega_b, ns, As, h])
    
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