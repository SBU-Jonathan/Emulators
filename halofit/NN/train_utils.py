# Auxiliary functions for Power Spectrum Emulation
# Author: João Victor Silva Rebouças, May 2022
# import camb
# from camb import model
import numpy as np
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy.fftpack import dst, idst
from scipy.integrate import simps
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from keras.regularizers import l1_l2
#------------------------------------------------------------------------------------------------------------
# Parameter space
params = ['h', 'Omegab', 'Omegam', 'As', 'ns', 'w']

# Parameter limits for training
lims = {}
lims['h'] = [0.61, 0.73]
lims['Omegab'] = [0.04, 0.06]
lims['Omegam'] = [0.24, 0.4]
lims['As'] = [1.7e-9, 2.5e-9]
lims['ns'] = [0.92, 1]
lims['w'] = [-1.3, -0.7]

# Parameter limits for testing - 10% tighter limits
test_lims = {}
test_lims['h'] = [0.62, 0.72]
test_lims['Omegab'] = [0.042, 0.058]
test_lims['Omegam'] = [0.255, 0.385]
test_lims['As'] = [1.78e-9, 2.42e-9]
test_lims['ns'] = [0.928, 0.992]

# Reference values
ref = {}
ref['h'] = 0.67
ref['Omegab'] = 0.049
ref['Omegam'] = 0.319
ref['As'] = 2.1e-9
ref['ns'] = 0.96
ref['w'] = -1

# Define k, z bins
redshifts = np.array([9.182, 7.690, 6.579, 5.720, 5.036, 4.478, 4.015, 3.624, 3.289, 3.000, 2.750, 2.529, 2.333, 2.158, 2.000, 1.824, 1.667, 1.526, 1.400, 1.286, 1.182, 1.087, 1.000, 0.929, 0.862, 0.800, 0.742, 0.688, 0.636, 0.588, 0.543, 0.500, 0.457, 0.417, 0.378, 0.342, 0.308, 0.275, 0.244, 0.214, 0.186, 0.159, 0.133, 0.109, 0.085, 0.062, 0.041, 0.020, 0.000])
redshifts = np.flip(redshifts)
ks_400 = np.genfromtxt('./power_spectra/lcdm/400/pk_ref_z_0.000.txt', unpack=True, usecols=(0))
ks_800 = np.genfromtxt('./power_spectra/lcdm/800/pk_ref_z_0.000.txt', unpack=True, usecols=(0))
#------------------------------------------------------------------------------------------------------------
def get_pk(h, Omegab, Omegam, As10to9, ns, w, redshifts = np.linspace(3, 0, 101), tau = 0.078):
    '''
    Returns [k, Pk_lin, Pk_nonlin], the scales and power spectrum for the given cosmology.
    k: array of scales in h/Mpc.
    Pk_lin: array of shape (len(redshifts), len(k)) of linear matter power spectra.
    Pk_nonlin: array of shape (len(redshifts), len(k)) of nonlinear matter power spectra.
    In the Pk arrays, latest redshift comes first.
    Set to 250 k-bins from k = 1e-4 to k = 10 (h/Mpc).
    '''
    Omegac = Omegam - Omegab
    As = As10to9 * 10**-9
    cosmology = camb.set_params(# Background
                                H0 = 100*h, ombh2=Omegab*h**2, omch2=Omegac*h**2,
                                TCMB = 2.7255,
                                # Dark Energy
                                dark_energy_model='fluid', w = w,
                                # Neutrinos
                                nnu=3.046, mnu = 0.058,
                                # Initial Power Spectrum
                                As = As, ns = ns, tau = tau,
                                YHe = 0.246, WantTransfer=True)
    cosmology.set_matter_power(redshifts=redshifts, kmax=100.0)
    cosmology.NonLinear = model.NonLinear_none
    results = camb.get_results(cosmology)
    
    # Calculating Linear Pk
    ks, _, pk_lin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=100, npoints = 1200)
    
    # Recalculating Pk with Nonlinear
    cosmology.NonLinear = model.NonLinear_both
    cosmology.NonLinearModel.set_params(halofit_version='takahashi')
    results.calc_power_spectra(cosmology)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-3, maxkh=100, npoints = 1200)   
    return ks, pk_lin, pk_nonlin
#------------------------------------------------------------------------------------------------------------
def normalize_array(array):
    '''
    Returns the Min-Max normalized array: (array - min(array))/(max(array) - min(array))
    '''
    return (array - np.amin(array))/(np.amax(array) - np.amin(array))

#------------------------------------------------------------------------------------------------------------
def unnormalize_array(norm_array, original_array):
    '''
    Returns `norm_array` scaled by `original_array`: norm_array * (max(array) - min(array)) + min(array)
    '''
    return norm_array * (np.amax(original_array) - np.amin(original_array)) + np.amin(original_array)
#------------------------------------------------------------------------------------------------------------
def normalize_params(params):
    '''
    Takes a params = (h, Omegam, Omegab, As10to9, ns, wde) array and returns normalized parameters.
    Normalization is given by normalized_param = (param - param_min)(param_max - param_min).
    '''
    if len(params) == 6:
        h, Omegab, Omegam, As10to9, ns, wde = params
        normalized_params = [(h-lims['h'][0])/(lims['h'][1] - lims['h'][0]),
                             (Omegab-lims['Omegab'][0])/(lims['Omegab'][1] - lims['Omegab'][0]),
                             (Omegam-lims['Omegam'][0])/(lims['Omegam'][1] - lims['Omegam'][0]),
                             (As10to9-lims['As'][0])/(lims['As'][1] - lims['As'][0]),
                             (ns-lims['ns'][0])/(lims['ns'][1] - lims['ns'][0]),
                             (wde-lims['w'][0])/(lims['w'][1] - lims['w'][0])]
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
#------------------------------------------------------------------------------------------------------------
def pca_reduction(num_pcs, norm_log_boosts, redshifts=redshifts):
    '''
    Fits a PCA under all redshifts and reduces the log boosts to their principal components
    Returns `pcas`, the set of PCAs for each redshift, and `reduced_norm_log_boosts`, the PCs of our input data.
    Also works on qs since it's the same process!
    '''
    pcas = []
    reduced_norm_log_boosts = np.zeros((len(norm_log_boosts), len(redshifts), num_pcs))
    for i in range(len(redshifts)):
        pca = PCA(n_components=num_pcs)
        reduced_norm_log_boosts_thisz = pca.fit_transform(norm_log_boosts[:,i,:])
        pcas.append(pca)
        reduced_norm_log_boosts[:,i,:] = reduced_norm_log_boosts_thisz
    return pcas, reduced_norm_log_boosts   
#------------------------------------------------------------------------------------------------------------
def pca_reconstruction_analysis(num_pcs, norm_qs, qs, sample, save=False, imgName='pc_analysis.pdf'):
    '''
    Plots the reconstruction errors for a given number of PCs
    The boosts can be replaced by Qs
    '''
    pcas = []
    reduced_norm_qs = np.zeros((len(sample), len(redshifts), num_pcs))
    reconstructed_boosts = np.zeros((len(sample), len(redshifts), len(ks)))
    for i in range(len(redshifts)):
        pca = PCA(n_components=num_pcs)
        reduced_norm_qs_thisz = pca.fit_transform(norm_qs[:,i,:])
        reconstructed_norm_qs_thisz = pca.inverse_transform(reduced_norm_qs_thisz)
        pcas.append(pca)
        reduced_norm_qs[:,i,:] = reduced_norm_qs_thisz
        reconstructed_qs_thisz = unnormalize_array(reconstructed_norm_qs_thisz, qs)
        reconstructed_boost_thisz = np.exp(reconstructed_qs_thisz)
        reconstructed_boosts[:,i,:] = reconstructed_boost_thisz
    # Plot reconstructed data
    for i, point in enumerate(sample):
        for j, redshift in enumerate(redshifts):
            plt.semilogx(ks, reconstructed_boosts[i,j,:]/np.exp(qs)[i,j,:] - 1, c='gray',alpha=0.5)
    plt.fill_between(ks, -0.01,0.01, color='gray',alpha=0.2)
    plt.grid()
    plt.xlabel(r'$k$ (h/Mpc)')
    which_quantity = input('Which is the emulated quantity (B or Q)? ')
    if which_quantity=='B':
        plt.ylabel(r'$B^{PCA}/B^{original} - 1$')
    if which_quantity=='Q':
        plt.ylabel(r'$B^{PCA}/B^{original} - 1$')
    title = input('Enter a title for this plot: ')
    plt.title(title)
    if save:
        plt.savefig(imgName)
    plt.show()
#------------------------------------------------------------------------------------------------------------
def emulate_boost(emulator, params, pcas, pc_components, log_boosts):
    '''
    Emulates the boosts using the NN model `emulator` for a set of params
    '''
    norm_params = normalize_params(params)
    emulated_norm_pcs = emulator(np.array([norm_params]))
    emulated_pcs = unnormalize_array(emulated_norm_pcs, pc_components)
    emulated_norm_log_boost = pcas[0].inverse_transform(emulated_pcs)
    emulated_log_boost = unnormalize_array(emulated_norm_log_boost, log_boosts)
    emulated_boost = np.exp(emulated_log_boost)
    return emulated_boost
#------------------------------------------------------------------------------------------------------------
def find_first_minimum(array):
    for i, entry in enumerate(array):
        if i <= 10:
            continue
        else:
            left_neighbor = array[i-1]
            right_neighbor = array[i+1]
            if entry < left_neighbor and entry < right_neighbor:
                return i, entry
        if i == len(array)-1:
            return 'Error'
def find_second_maximum(array):
    maxima = 0
    for i, entry in enumerate(array):
        if i <= 10:
            continue
        else:
            left_neighbor = array[i-1]
            right_neighbor = array[i+1]
            if entry > left_neighbor and entry > right_neighbor:
                maxima += 1
                if maxima == 2:
                    return i, entry
        if i == len(array) - 1:
            return 'Error'
def smooth_bao(ks, pk):
    spline_loglog_pk = CubicSpline(np.log(ks), np.log(pk))
    n = 10
    dst_ks = np.linspace(ks[0], ks[-1], 2**n)
    logks = np.log(dst_ks)
    logkpk = logks + spline_loglog_pk(logks)
    sine_transf_logkpk = dst(logkpk, type=2, norm='ortho')
    odds = [] # odd entries
    evens = [] # even entries
    even_is = [] # odd indices
    odd_is = [] # even indices
    all_is = [] # all indices
    for i, entry in enumerate(sine_transf_logkpk):
        all_is.append(i)
        if i%2 == 0:
            even_is.append(i)
            evens.append(entry)
        else:
            odd_is.append(i)
            odds.append(entry)
    odd_is = np.array(odd_is)
    even_is = np.array(even_is)
    odds_interp = CubicSpline(odd_is, odds)
    evens_interp = CubicSpline(even_is, evens)
    d2_odds = (odds_interp.derivative(nu=2))
    d2_evens = (evens_interp.derivative(nu=2))
    d2_odds_avg = (d2_odds(odd_is) + d2_odds(odd_is + 2) + d2_odds(odd_is - 2))/3
    d2_evens_avg = (d2_evens(even_is) + d2_evens(even_is + 2) + d2_evens(even_is - 2))/3
    i_star_bottom, _ = find_first_minimum(d2_odds_avg)
    i_star_top, _ = find_second_maximum(d2_odds_avg)
    imin_odd = i_star_bottom - 3
    imax_odd = i_star_top + 20
    i_star_bottom, _ = find_first_minimum(d2_evens_avg)
    i_star_top, _ = find_second_maximum(d2_evens_avg)
    imin_even = i_star_bottom - 3
    imax_even = i_star_top + 10   
    odd_is_removed_bumps = []
    odds_removed_bumps = []
    for i, entry in enumerate(odds):
        if i in range(imin_odd, imax_odd+1):
            continue
        else:
            odd_is_removed_bumps.append(2*i+1)
            odds_removed_bumps.append(entry)
    even_is_removed_bumps = []
    evens_removed_bumps = []
    for i, entry in enumerate(evens):
        if i in range(imin_even, imax_even+1):
            continue
        else:
            even_is_removed_bumps.append(2*i)
            evens_removed_bumps.append(entry)
    odd_is_removed_bumps = np.array(odd_is_removed_bumps)
    even_is_removed_bumps = np.array(even_is_removed_bumps)
    odds_removed_spline_iplus1 = CubicSpline(odd_is_removed_bumps, (odd_is_removed_bumps+1)**2 * odds_removed_bumps)
    evens_removed_spline_iplus1 = CubicSpline(even_is_removed_bumps, (even_is_removed_bumps+1)**2 * evens_removed_bumps)
    odds_treated_iplus1 = odds_removed_spline_iplus1(odd_is)
    evens_treated_iplus1 = evens_removed_spline_iplus1(even_is)
    odds_treated = odds_treated_iplus1/(odd_is+1)**2
    evens_treated = evens_treated_iplus1/(even_is+1)**2
    treated_transform = []
    for odd, even in zip(odds_treated, evens_treated):
        treated_transform.append(even)
        treated_transform.append(odd)
    treated_logkpk = idst(treated_transform, type=2, norm='ortho')
    pk_nw = np.exp(treated_logkpk)/dst_ks
    pk_nw_spline = CubicSpline(dst_ks, pk_nw)
    pk_nw = pk_nw_spline(ks)
    return pk_nw
def smear_bao(ks, pk, pk_nw):
    integral = simps(pk,ks)
    k_star_inv = (1.0/(3.0 * np.pi**2)) * integral
    Gk = np.array([np.exp(-0.5*k_star_inv * (k_**2)) for k_ in ks])
    pk_smeared = pk*Gk + pk_nw*(1.0 - Gk)
    return pk_smeared
#------------------------------------------------------------------------------------------------------------
def generate_and_save_train_spectra(sample):
    '''
    Generates linear and nonlinear power spectra for the LHS points in `sample`
    Input: LHS sample in the format (h, Omegab, Omegam, As10to9, ns, wde)
    Output: pks_lin and pk_nonlin, arrays of power spectra pk(k, z) for each sample point for the specified k-bins and redshifts
    '''
    pks_lin = np.zeros((len(sample), len(redshifts), len(ks)))
    pks_nonlin = np.zeros((len(sample), len(redshifts), len(ks)))
    for i, point in enumerate(sample):
        h_point, Omegab_point, Omegam_point, As10to9_point, ns_point, wde_point = point
        _, pk_lin_point, pk_nonlin_point = get_pk(h_point, Omegab_point, Omegam_point, As10to9_point, ns_point, wde_point)
        pks_lin[i] = pk_lin_point
        pks_nonlin[i] = pk_nonlin_point
        if i < 10:
            np.savetxt('train_pks/lin_00'+str(i)+'.txt', pks_lin[i])
            np.savetxt('train_pks/nonlin_00'+str(i)+'.txt', pks_nonlin[i])
        elif i < 100:
            np.savetxt('train_pks/lin_0'+str(i)+'.txt', pks_lin[i])
            np.savetxt('train_pks/nonlin_0'+str(i)+'.txt', pks_nonlin[i])
        else:
            np.savetxt('train_pks/lin_'+str(i)+'.txt', pks_lin[i])
            np.savetxt('train_pks/nonlin_'+str(i)+'.txt', pks_nonlin[i])
        fraction = (i+1)/len(sample)
        # Reporting progress
        if fraction in np.arange(0.1, 1.1, 0.1):
            percentage = fraction*100
            num_of_dashes = round(fraction*10)
            progress_bar = '[' + '-'*num_of_dashes + ' '*(10-num_of_dashes) + ']'
            print('Progress: ', percentage, '% ', progress_bar)
    return pks_lin, pks_nonlin
#------------------------------------------------------------------------------------------------------------
def generate_and_save_test_spectra(sample):
    '''
    Generates test linear and nonlinear power spectra for k and z bins determined in redshifts and ks
    '''
    test_pks_lin = np.zeros((len(sample), len(redshifts), len(ks)))
    test_pks_nonlin = np.zeros((len(sample), len(redshifts), len(ks)))
    for i, point in enumerate(sample):
        h_point, Omegab_point, Omegam_point, As10to9_point, ns_point, wde_point = point
        _, pk_lin_point, pk_nonlin_point = get_pk(h_point, Omegab_point, Omegam_point, As10to9_point, ns_point, wde_point)
        test_pks_lin[i] = pk_lin_point
        test_pks_nonlin[i] = pk_nonlin_point
        if i < 10:
            np.savetxt('test_pks/lin_00'+str(i)+'.txt', test_pks_lin[i])
            np.savetxt('test_pks/nonlin_00'+str(i)+'.txt', test_pks_nonlin[i])
        elif i < 100:
            np.savetxt('test_pks/lin_0'+str(i)+'.txt', test_pks_lin[i])
            np.savetxt('test_pks/nonlin_0'+str(i)+'.txt', test_pks_nonlin[i])
        else:
            np.savetxt('test_pks/lin_'+str(i)+'.txt', test_pks_lin[i])
            np.savetxt('test_pks/nonlin_'+str(i)+'.txt', test_pks_nonlin[i])
        fraction = (i+1)/len(sample)
        # Reporting progress
        if fraction in np.arange(0.1, 1.1, 0.1):
            percentage = fraction*100
            num_of_dashes = round(fraction*10)
            progress_bar = '[' + '-'*num_of_dashes + ' '*(10-num_of_dashes) + ']'
            print('Progress: ', percentage, '% ', progress_bar)
    return test_pks_lin, test_pks_nonlin
#------------------------------------------------------------------------------------------------------------
def load_spectra_from_files(sample, dataset):
    assert (dataset=='train' or dataset=='test'), 'Dataset must be one of \'train\' or \'test\''
    folder=dataset+'_pks'
    pks_lin = np.zeros((len(sample), len(redshifts), len(ks)))
    pks_nonlin = np.zeros((len(sample), len(redshifts), len(ks)))
    for i, point in enumerate(sample):
        if i < 10:
            pks_lin[i] = np.loadtxt(folder+'/lin_00'+str(i)+'.txt')
            pks_nonlin[i] = np.loadtxt(folder+'/nonlin_00'+str(i)+'.txt')
        elif i < 100:
            pks_lin[i] = np.loadtxt(folder+'/lin_0'+str(i)+'.txt')
            pks_nonlin[i] = np.loadtxt(folder+'/nonlin_0'+str(i)+'.txt')
        else:
            pks_lin[i] = np.loadtxt(folder+'/lin_'+str(i)+'.txt')
            pks_nonlin[i] = np.loadtxt(folder+'/nonlin_'+str(i)+'.txt')
    return pks_lin, pks_nonlin
#------------------------------------------------------------------------------------------------------------
def generate_and_save_smear_bao(pks_lin, dataset):
    '''
    Takes an array of linear power spectra and returns the smeared version
    '''
    assert (dataset=='train' or dataset=='test'), 'Dataset must be one of \'train\' or \'test\''
    pks_lin_smear = np.zeros((len(pks_lin), len(redshifts), len(ks)))
    folder=dataset+'_pks'
    for i, pk_lin in enumerate(pks_lin):
        for j, z in enumerate(redshifts):
            pks_lin_smear[i, j] = smear_bao(ks, pk_lin[j])
        if i < 10:
            np.savetxt(folder+'/smear_00'+str(i)+'.txt', pks_lin_smear[i])
        elif i < 100:
            np.savetxt(folder+'/smear_0'+str(i)+'.txt', pks_lin_smear[i])
        else:
            np.savetxt(folder+'/smear_'+str(i)+'.txt', pks_lin_smear[i])
    return pks_lin_smear
#------------------------------------------------------------------------------------------------------------
def load_smear_bao(sample, dataset):
    assert (dataset=='train' or dataset=='test'), 'Dataset must be one of \'train\' or \'test\''
    folder=dataset+'_pks'
    pks_smear = np.zeros((len(sample), len(redshifts), len(ks)))
    for i, point in enumerate(sample):
        if i < 10:
            pks_smear[i] = np.loadtxt(folder+'/smear_00'+str(i)+'.txt')
        elif i < 100:
            pks_smear[i] = np.loadtxt(folder+'/smear_0'+str(i)+'.txt')
        else:
            pks_smear[i] = np.loadtxt(folder+'/smear_'+str(i)+'.txt')
    return pks_smear
#------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------
def generate_model(num_pcs, num_of_layers, input_data, num_of_neurons=512, activation="relu"):
    '''
    Returns a keras sequential NN model with `num_of_layers` hidden layers with `num_of_neurons` each.
    '''
    nn_layers = []
    # Adding layers
    input_layer = Input(shape=len(input_data[0]))
    nn_layers.append(input_layer)
    for i in range(num_of_layers):
        nn_layers.append(layers.Dense(num_of_neurons, activation=activation, name='hid_layer'+str(i+1)))
        nn_layers.append(layers.BatchNormalization())
    # Output layer has `num_pcs` neurons
    nn_layers.append(layers.Dense(num_pcs, activation=activation, name='out_layer'))
    model = keras.Sequential(nn_layers)
    y = model(np.array([input_data[0]])) # Calling model on an example input to initialize the input layer
    model.summary()
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanAbsoluteError()
    )
    return model
#------------------------------------------------------------------------------------------------------------
def generate_model_regularized(num_pcs, num_of_layers, input_data, num_of_neurons=512, activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
    '''
    Returns a keras sequential NN model with `num_of_layers` hidden layers with `num_of_neurons` each.
    '''
    nn_layers = []
    regularization_term = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
    # Adding layers
    input_layer = layers.Input(shape=len(input_data[0]))
    nn_layers.append(input_layer)
    for i in range(num_of_layers):
        nn_layers.append(layers.Dense(num_of_neurons, activation=activation, name=f'hid_layer_{i+1}', 
            kernel_regularizer=regularization_term, bias_regularizer=regularization_term))
        if dropout != 0:
            nn_layers.append(layers.Dropout(dropout, name=f'dropout_{i+1}'))
        #nn_layers.append(layers.BatchNormalization())
    # Output layer has `num_pcs` neurons
    out_layer = layers.Dense(num_pcs, activation=activation, name='out_layer')
    nn_layers.append(out_layer)
    model = keras.Sequential(nn_layers)
    model.summary()
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanAbsoluteError()
    )
    return model
#------------------------------------------------------------------------------------------------------------
def generate_resnet(num_pcs, num_res_blocks, input_data, num_of_neurons=512, activation="relu", alpha=1e-5, l1_ratio=0.1, dropout=0.1):
    '''
    Generates a ResNet model with `num_res_blocks` residual blocks.
    '''
    nn_layers = []
    regularization_term = l1_l2(l1=alpha*l1_ratio, l2=alpha*(1-l1_ratio))
    
    # Adding layers
    input_layer = layers.Input(shape=len(input_data[0]))
    
    # Adding first residual block
    hid1 = layers.Dense(units=num_of_neurons,
         activation='relu',
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(input_layer)

    hid2 = layers.Dense(units=num_of_neurons,
         activation='relu',
         kernel_regularizer=regularization_term,
         bias_regularizer=regularization_term)(hid1)
    
    residual = layers.Add()([hid1, hid2])
    
    if num_res_blocks > 1:
        for i in range(num_res_blocks - 1):
            hid1 = layers.Dense(units=num_of_neurons,
                 activation='relu',
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(residual)

            hid2 = layers.Dense(units=num_of_neurons,
                 activation='relu',
                 kernel_regularizer=regularization_term,
                 bias_regularizer=regularization_term)(hid1)

            residual = layers.Add()([hid1, hid2])
    
    output_layer = layers.Dense(units=num_pcs, activation='relu')(residual)
    
    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    model.summary()
    
    model.compile(
        optimizer = keras.optimizers.Adam(),
        loss = keras.losses.MeanAbsoluteError()
    )
    
    return model
#------------------------------------------------------------------------------------------------------------
def test_model(emulator, test_sample, test_data, pc_components, pca_z, log_quantity, save=False):
    '''
    Plots emulation errors for the test data and returns the max errors of the test data
    '''
    for i, test_quantity in enumerate(test_data):
        norm_test_params = normalize_params(test_sample[i])
        emulated_test_norm_pcs = emulator(np.array([norm_test_params]))
        emulated_test_pcs = unnormalize_array(emulated_test_norm_pcs, pc_components)
        emulated_test_norm_log_quantity = pca_z.inverse_transform(emulated_test_pcs)
        emulated_test_log_quantity = unnormalize_array(emulated_test_norm_log_quantity, log_quantity)
        emulated_test_quantity = np.exp(emulated_test_log_quantity)
        plt.semilogx(ks, emulated_test_quantity[0]/test_quantity[0] - 1, c='gray', alpha=0.5)
    
    plt.grid()
    plt.fill_between(ks, -0.01,0.01, color='gray',alpha=0.2)
    plt.xlabel('k (h/Mpc)')
    plt.ylabel('Emulation Relative Error')
    title = input('Define a title for the plot: ')
    plt.title(title)
    if save:
        imgName = input('Define the archive name: ')
        plt.savefig(imgName)
    plt.show()
#------------------------------------------------------------------------------------------------------------
def evaluate_test_points(emulator, test_sample, test_data, pc_components, pca_z, log_quantity):
    '''
    Returns max_emulation_errors, the maximum emulation errors for the test data
    '''
    max_emulation_errors = np.zeros(len(test_sample))
    for i, test_quantity in enumerate(test_data):
        norm_test_params = normalize_params(test_sample[i])
        emulated_test_norm_pcs = emulator(np.array([norm_test_params]))
        emulated_test_pcs = unnormalize_array(emulated_test_norm_pcs, pc_components)
        emulated_test_norm_log_quantity = pca_z.inverse_transform(emulated_test_pcs)
        emulated_test_log_quantity = unnormalize_array(emulated_test_norm_log_quantity, log_quantity)
        emulated_test_quantity = np.exp(emulated_test_log_quantity)
        max_emulation_errors[i] = np.amax(emulated_test_quantity[0]/test_quantity[0] - 1)
    return max_emulation_errors
#------------------------------------------------------------------------------------------------------------
def iterative_training(emulator, initial_training_sample, initial_training_data, initial_test_sample, initial_test_data):
    assert False, 'Not implemented'
    # Initial training
    history = emulator_model.fit(
            initial_training_sample,
            initial_training_data,
            batch_size = 5,
            epochs = 2000,
            # validation_data = (validation_input_data, validation_truth_data)
            )
    emulation_errors = evaluate_test_points(emulator, initial_test_sample, initial_test_data, pc_components, pca_z, log_quantity)
    sorted_errors = np.sort(emulation_errors, order='descending')
    treshold = sorted_errors[round(len(emulation_errors)/5)] # The top 20% error
    outliers = []
    outliers_data = []
    for i, error in enumerate(emulation_errors):
        if error > treshold:
            outliers.append(initial_test_sample[i])
            outliers_data.append(initial_test_data[i])
    outliers = np.array(outliers)
    outliers_data = np.array(outliers_data)
    new_training_sample = np.concatenate((initial_training_sample, outliers))
    new_training_data = np.concatenate((initial_traiing_data, outliers_data))

#------------------------------------------------------------------------------------------------------------ 
def _nowiggles_pk(k_lin=None, pk_lin=None, k_emu=None):
    """De-wiggled linear prediction of the cold matter power spectrum

    The BAO feature is removed by identifying and removing its corresponding
    bump in real space, by means of a DST, and consequently transforming
    back to Fourier space.
    See:
    - Baumann et al 2018 (https://arxiv.org/pdf/1712.08067.pdf)
    - Giblin et al 2019 (https://arxiv.org/pdf/1906.02742.pdf)

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional

    :return: dewiggled pk computed at k_emu
    :rtype: array_like
    """

    from scipy.fftpack import dst, idst

    nk = int(2**15)
    kmin = k_lin.min()
    kmax = 10
    klin = np.linspace(kmin, kmax, nk)

    pkcamb_cs = interpolate.splrep(np.log(k_lin), np.log(pk_lin), s=0)
    pklin = np.exp(interpolate.splev(np.log(klin), pkcamb_cs, der=0, ext=0))

    f = np.log10(klin * pklin)

    dstpk = dst(f, type=2)

    even = dstpk[0::2]
    odd = dstpk[1::2]

    i_even = np.arange(len(even)).astype(int)
    i_odd = np.arange(len(odd)).astype(int)

    even_cs = interpolate.splrep(i_even, even, s=0)
    odd_cs = interpolate.splrep(i_odd, odd, s=0)

    even_2nd_der = interpolate.splev(i_even, even_cs, der=2, ext=0)
    odd_2nd_der = interpolate.splev(i_odd, odd_cs, der=2, ext=0)

    # these indexes have been fudged for the k-range considered
    # [~1e-4, 10], any other choice would require visual inspection
    imin_even = i_even[100:300][np.argmax(even_2nd_der[100:300])] - 20
    imax_even = i_even[100:300][np.argmin(even_2nd_der[100:300])] + 70
    imin_odd = i_odd[100:300][np.argmax(odd_2nd_der[100:300])] - 20
    imax_odd = i_odd[100:300][np.argmin(odd_2nd_der[100:300])] + 75

    i_even_holed = np.concatenate((i_even[:imin_even], i_even[imax_even:]))
    i_odd_holed = np.concatenate((i_odd[:imin_odd], i_odd[imax_odd:]))

    even_holed = np.concatenate((even[:imin_even], even[imax_even:]))
    odd_holed = np.concatenate((odd[:imin_odd], odd[imax_odd:]))

    even_holed_cs = interpolate.splrep(i_even_holed, even_holed * (i_even_holed+1)**2, s=0)
    odd_holed_cs = interpolate.splrep(i_odd_holed, odd_holed * (i_odd_holed+1)**2, s=0)

    even_smooth = interpolate.splev(i_even, even_holed_cs, der=0, ext=0) / (i_even + 1)**2
    odd_smooth = interpolate.splev(i_odd, odd_holed_cs, der=0, ext=0) / (i_odd + 1)**2

    dstpk_smooth = []
    for ii in range(len(i_even)):
        dstpk_smooth.append(even_smooth[ii])
        dstpk_smooth.append(odd_smooth[ii])
    dstpk_smooth = np.array(dstpk_smooth)

    pksmooth = idst(dstpk_smooth, type=2) / (2 * len(dstpk_smooth))
    pksmooth = 10**(pksmooth) / klin

    k_highk = k_lin[k_lin > 5]
    p_highk = pk_lin[k_lin > 5]

    k_extended = np.concatenate((klin[klin < 5], k_highk))
    p_extended = np.concatenate((pksmooth[klin < 5], p_highk))

    pksmooth_cs = interpolate.splrep(np.log(k_extended), np.log(p_extended), s=0)
    pksmooth_interp = np.exp(interpolate.splev(np.log(k_emu), pksmooth_cs, der=0, ext=0))

    return pksmooth_interp

def _smeared_bao_pk(k_lin=None, pk_lin=None, k_emu=None, pk_lin_emu=None, pk_nw=None, grid=None):
    """Prediction of the cold matter power spectrum using a Boltzmann solver with smeared BAO feature

    :param k_lin: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              camb are returned, defaults to None
    :type k_lin: array_like, optional
    :param pk_lin: a vector of linear power spectrum computed at k_lin, if None
              camb will be called, defaults to None
    :type pk_lin: array_like, optional

    :param k_emu: a vector of wavemodes in h/Mpc, if None the wavemodes used by
              the emulator are returned, defaults to None
    :type k_emu: array_like, optional
    :param pk_emu: a vector of linear power spectrum computed at k_emu, defaults to None
    :type pk_emu: array_like, optional
    :param pk_nw: a vector of no-wiggles power spectrum computed at k_emu, defaults to None
    :type pk_nw: array_like, optional
    :param grid: dictionary with parameter and vector of values where to evaluate the emulator, defaults to None
    :type grid: array_like, optional

    :return: smeared BAO pk computed at k_emu
    :rtype: array_like
    """
    from scipy.integrate import trapz

    if grid is None:
        sigma_star_2 = trapz(k_lin * pk_lin, x=np.log(k_lin)) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2))
        if pk_nw is None:
            pk_nw = _nowiggles_pk(k_lin=k_lin, pk_lin=pk_lin, k_emu=k_emu)
    else:
        sigma_star_2 = trapz(k_lin[None,:] * pk_lin, x=np.log(k_lin[None:,]), axis=1) / (3 * np.pi**2)
        k_star_2 = 1 / sigma_star_2
        G = np.exp(-0.5 * (k_emu**2 / k_star_2[:,None]))
        if pk_nw is None:
            pk_nw = np.array([_nowiggles_pk(k_lin=k_lin, pk_lin=pkl, k_emu=k_emu) for pkl in pk_lin])
    return pk_lin_emu * G + pk_nw * (1 - G)

def nn_model_train(model, epochs, input_data, truths, validation_features=None, validation_truths=None, decayevery=None, decayrate=None):
    '''
    Creates and trains a neural network model that emulates the pc_components from the input_data
    Returns
    '''
    if decayevery and decayrate: 
        def scheduler(epoch, learning_rate):
            # Halves the learning rate at some points during training
            if epoch != 0 and epoch % decayevery == 0:
                return learning_rate/decayrate
            else:
                return learning_rate
    
    if decayevery:
        learning_scheduler = keras.callbacks.LearningRateScheduler(scheduler)
    else:
        learning_scheduler = keras.callbacks.LearningRateScheduler(lambda epoch, learning_rate: learning_rate)
    
    if validation_features and validation_truths:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            validation_data = (validation_features, validation_truths),
            callbacks=[learning_scheduler],
        )
    else:
        history = model.fit(
            input_data,
            truths,
            batch_size = 30,
            epochs = epochs,
            callbacks=[learning_scheduler],
        )
    
    last_loss = history.history['loss'][-1]
    return last_loss