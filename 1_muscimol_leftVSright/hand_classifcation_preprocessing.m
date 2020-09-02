%% Add Paths
% restoredefaultpath
addpath('D:\dancause_data\DancauseData2019')
addpath('D:\dancause_data\DancauseData2019\LFP_data')
addpath('D:\dancause_data\DancauseData2019\dataNuma10952020')
addpath('D:\dancause_data\utils')
%% Load Data
% load('2014_10_14_G_Pre.mat') % already exported
load("2014_10_14_G_T00.mat");
%% Save path
savepath = 'D:\dancause_data\cleaned_data';
%% Hyperparameters

save_name = 'subject1_post.mat';
minimum_grasp_time = 500; % time in ms;
subsample_size = 1000; % samples to take from middle of grasp action %% TEST W/ THIS SUBSAMPLE_SIZE! ONLY 100 NOW! must be less than (minimum_grasp_time / 1000 * fs)
n_stds = 5; % n stds to include, outside of this range considered outliers, replaced with NaN
cutoff.lower = 5; cutoff.upper = 100; % bandpass upper and lower cutoff
cleaningSettings.switches = [0,1,1,1]; %[bandpass, de-mean, NaN outliers, interpolate nans] booleans to choose which cleaning steps to perform

initialize_useful_variables;
%% Cleaning

lfpData = cleanlfpData(lfpData, cleaningSettings, fs);

%% Reshape lfpData: only grasp samples, trials

lfpData = reshapeTrials(lfpData, lfpParam, taskParam, subsample_size, minimum_grasp_time, fs);

%% Reshape lfpData: left hand, right hand

lfpData = reshapeHands(lfpData, taskParam, minimum_grasp_time);

%% Save preprocessed data

dataSorted = lfpData; 
cd(savepath)
save(save_name, 'dataSorted') % output format: dataSorted = n_channels x time_samples x trials x blocks x sessions







