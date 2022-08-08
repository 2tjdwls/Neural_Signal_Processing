"""
Created on Mon Aug  8 14:11:33 2022

@ Title : Source Estimation
@ author: dltjdwls

"""

import os # PATH 관련 툴
import mne # MNE 1.0.0
import numpy as np # 수학적 계산을 위한 툴
import mat73

cwd = os.getcwd() # 현재 작업 경로 가져오기
print('$ Print MNE version:   ', mne.__version__)
print('$ Current working dir:   ', cwd)

# Read EEG_mat file
os.chdir('C:/Users/tjdwl/Desktop/인간뇌기능 연구실/첫번째 연구/EEG memory data/01 jungchaeho')
cwd = os.getcwd()
print('$ Current working dir:   ', cwd)

false1 = mat73.loadmat('false1.mat')
false2 = mat73.loadmat('false2.mat')
false3 = mat73.loadmat('false3.mat')

true1 = mat73.loadmat('true1.mat')
true2 = mat73.loadmat('true2.mat')
true3 = mat73.loadmat('true3.mat')

# Converting from MAT to Numpy array
false1 = false1['false_epoch']
false2 = false2['false_epoch']
false3 = false3['false_epoch']

true1 = true1['true_epoch']
true2 = true2['true_epoch']
true3 = true3['true_epoch']

# Channels x time points x trials
false1.shape, false2.shape, false3.shape, true1.shape, true2.shape, true3.shape

false1.shape[2] # trials

# Delete CB1, CB2 channel

false1 = np.delete(false1, 56, 0)
false1 = np.delete(false1, 59, 0)
false2 = np.delete(false2, 56, 0)
false2 = np.delete(false2, 59, 0)
false3 = np.delete(false3, 56, 0)
false3 = np.delete(false3, 59, 0)

true1 = np.delete(true1, 56, 0)
true1 = np.delete(true1, 59, 0)
true2 = np.delete(true2, 56, 0)
true2 = np.delete(true2, 59, 0)
true3 = np.delete(true3, 56, 0)
true3 = np.delete(true3, 59, 0)

# Data 처리하기 편하게 dictionary로 묶기

eeg_list = [false1, false2, false3, true1, true2, true3]
key = ['F1', 'F2', 'F3', 'T1', 'T2', 'T3']
eeg_dict = dict()

for e,k in zip(eeg_list, key):
    eeg_dict[k] = e
    

eeg_dict['F1'].shape

# Channel name
# Rejection channel : Cp1
ch_names = ['Fp1', 'Fpz', 'Fp2',
           'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
           'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8',
           'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8',
           'TP7', 'CP5', 'CP3', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8',
           'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
           'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8',
           'O1', 'Oz', 'O2']

len(ch_names)

# MNE tutorials
n_channels = 59
sampling_freq = 1000
ch_types = ['eeg'] * 59
info = mne.create_info(ch_names, ch_types = ch_types, sfreq=sampling_freq)

info.set_montage('standard_1005')

eeg_dict['F1']

for i in key:
    eeg_dict[i] = eeg_dict[i].swapaxes(1,2)
    eeg_dict[i] = eeg_dict[i].swapaxes(0,1)
    
eeg_dict['F1'].shape, eeg_dict['F2'].shape, eeg_dict['F3'].shape
eeg_dict['T1'].shape, eeg_dict['T2'].shape, eeg_dict['T3'].shape

# MNE epoch으로 변환

epochs_dict = dict()

for i in key:
    epochs_dict[i] = mne.EpochsArray(eeg_dict[i], info)
    
# MNE epoch plot

epochs_dict['T2'].plot(scalings=20, n_channels=60)

# Fetch freesurfer (Only first time)

fs_dir = mne.datasets.fetch_fsaverage(verbose=False)
subjects_dir = os.path.dirname(fs_dir)
print(subjects_dir)

# Load freesurfer average

subject = 'fsaverage'
trans = 'fsaverage'

src = mne.source_space.setup_source_space(
    subject=subject, subjects_dir = subjects_dir, # freesurfer
    surface = 'pial', # or white
    spacing = 'oct6', # add_dist = sparse than 'ico5' space
    add_dist = False
) # Takes a little time...

bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif')

# Read and set the EEG electrode location, which are already in faverage's space
# (MNI space) for standard_1005 :

# key = ['F1', 'F2', 'F3', 'T1', 'T2', 'T3']

montage = mne.channels.make_standard_montage('standard_1005')
epochs_dict['F1'].set_montage(montage)

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    epochs_dict['F1'].info, eeg=['original', 'projected'], trans=trans,
    show_axes=True, mri_fiducials=True, dig='fiducials', subject=subject, bem=bem)

# Check that the locations of EEG electrodes is correct with respect to MRI
mne.viz.plot_alignment(
    epochs_dict[key[0]].info, 
    src=src, 
    eeg='projected', 
    trans=trans,
    show_axes=True,
    mri_fiducials=True, 
    dig='fiducials',
    subject=subject,
    surfaces='head-dense',
    coord_frame='mri',
    bem=bem
)

# Compute forward solution
fwd = mne.make_forward_solution(epochs_dict[key[0]].info, trans=trans, src=src,
                                bem=bem, eeg=True, mindist=5.0, n_jobs=1)

# Compute noise covariance during baseline
noise_cov = mne.compute_covariance(epochs_dict[key[0]], tmax=0, method='auto')

# Inverse operator
inv = mne.minimum_norm.make_inverse_operator(
    info=epochs_dict[key[0]].info, 
    forward=fwd, noise_cov=noise_cov
)

mne.viz.plot_cov(noise_cov,info=epochs_dict[key[0]].info)

# Using the same inverse operator when inspecting single trials 
snr = 3.0  # Standard assumption for average data but using it for single trial
lambda2 = 1.0 / snr ** 2
# Apply inverse operator on single-trial epoch
stcs = mne.minimum_norm.apply_inverse_epochs(
    epochs=epochs_dict[key[0]], 
    inverse_operator=inv,
    lambda2=lambda2,
    method='dSPM'
)

# Averaging
mean_stcs = sum(stcs)/len(stcs)
mean_stcs = mean_stcs.crop(0, 5)

# plot the source!
mean_stcs.apply_baseline((0, 2))
mean_stcs.plot(
    surface='pial',
    cortex='low_contrast',
    src=src,
    hemi='both'
)

mean_stcs.save('mean-stcs', overwrite=True)
