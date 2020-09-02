%% Data path
save_name = 'dataSorted.mat';
datapath = 'D:\ub_neuroComp\dancause_data\stroke\stroke_data\20180608Y';
savepath = 'D:\ub_neuroComp\dancause_data\processing\3_prestroke_actionsANDorientations\export';
%% Add Paths
% restoredefaultpath
addpath(datapath)
addpath('D:\ub_neuroComp\dancause_data\processing\3_prestroke_actionsANDorientations')
addpath('D:\ub_neuroComp\dancause_data\processing\3_prestroke_actionsANDorientations\utils')
addpath(savepath)
%% Detect and select files
hand = {'Right'}; %'Left'
precision_angle = {'Precision_0'}; %,'45','90','135'};
aligned_to = {'GraspStart'}; %'CueOn'
spikes = {'spikeFree'}; % 
% data_options = [hand, aligned_to, spikes];

excluded_channels = struct; excluded_channels.channelFlags = [];
channel_ids = struct; channel_ids.channels= [];

all_files = dir(datapath);
file_names = cell(1,length(all_files));
for i = 1:length(file_names)
   file_names{i} = all_files(i).name; 
end

if strcmp(spikes{1}, 'spikeFree')
    inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle) & contains(file_names, spikes); 
else
    inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle) & ~contains(file_names, 'spikeFree');
end

files_to_load_0 = file_names(inds);

hand = {'Right'}; % , 'Left'
precision_angle = {'135'};%,'45','90','135'};
aligned_to = {'GraspStart'}; %'CueOn'
spikes = {'spikeFree'}; % 
% data_options = [hand, aligned_to, spikes];

all_files = dir(datapath);
file_names = cell(1,length(all_files));
for i = 1:length(file_names)
   file_names{i} = all_files(i).name; 
end

if strcmp(spikes{1}, 'spikeFree')
    inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle) & contains(file_names, spikes); 
else
    inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle);
end

files_to_load_135 = file_names(inds);
%% Load Data
% load('2014_10_14_G_Pre.mat') % already exported
% load("2014_10_14_G_T00.mat");
n_trials = 35;
n_orientations = 2;
load_0_degree_data;
excluded_channels(1).channelFlags = channelFlags_0;
load_135_degree_data;
excluded_channels(2).channelFlags = channelFlags_135;

load([datapath,'\channel.mat'])
channel_ids(1).channels = Channel;
%% Normalize
if 1 
    for t = 1:size(lfpData_0,3)
        for ch = 1:size(lfpData_0,1)
            mu = mean(lfpData_0(ch,:,t));
            sdev = std(lfpData_0(ch,:,t));
            lfpData_0(ch,:,t) = (lfpData_0(ch,:,t) - mu) / (sdev+0.00001);
        end
    end
    fprintf('0 Data normalized...\n')
    
    for t = 1:size(lfpData_135,3)
        for ch = 1:size(lfpData_135,1)
            mu = mean(lfpData_135(ch,:,t));
            sdev = std(lfpData_135(ch,:,t));
            lfpData_135(ch,:,t) = (lfpData_135(ch,:,t) - mu) / (sdev+0.00001);
        end
    end
    fprintf('135 Data normalized...\n')
end
%% Seperate actions

sample_duration = 0.175; % had to reduce from 0.25 due to min_reach

min_baseline = min([min(cueOns_0+Time(end)),min(cueOns_135+Time(end))]); % check sample duration
min_pre_grasp = min([min(cueOffs_0 - cueOns_0),min(cueOffs_135 - cueOns_135)]);
min_reach = min([min(graspStarts_0 - cueOffs_0),min(graspStarts_135 - cueOffs_135)]);
% min_grasp % grasp set to graspStart + sample_duration 
min_post_grasp = min([min(Time(end) - (graspStarts_0 + sample_duration)),min(Time(end) - (graspStarts_135 + sample_duration))]);
if  min_baseline < sample_duration ||  min_pre_grasp < sample_duration || min_reach < sample_duration ||  min_post_grasp < sample_duration
    error('Sample duration too large. Cannot make samples with identical durations.')
end

trial_duration = Time(end) - Time(1);
fs = length(Time)  / trial_duration;
samples_per_sample = floor(sample_duration * fs);

extract_actions_0;
extract_actions_135;



%% Join actions
out = cat(4, baselines_0, pre_grasps_0, reaches_0, grasps_0, post_grasps_0, baselines_135, pre_grasps_135, reaches_135, grasps_135, post_grasps_135);
clear baselines_0
clear pre_grasps_0
clear reaches_0
clear grasps_0
clear post_grasps_0
clear baselines_135
clear pre_grasps_135
clear reaches_135
clear grasps_135
clear post_grasps_135
%% Rearrange channels (hardcoded)
% ch_names = cell(size(channel_ids(1).channels,2),size(channel_ids,2));
% ch_groups = cell(size(channel_ids(1).channels,2),size(channel_ids,2));
% for cond = 1:size(channel_ids,2)
%     for ch = 1:size(channel_ids(1).channels,2)
%         ch_names{ch,cond}=channel_ids(cond).channels(ch).Name;
%         ch_groups{ch,cond}=channel_ids(cond).channels(ch).Group;
%     end
% end
% 
% left_PMd_pre = out(113:144,:,:,1:10); 
% left_PMv_pre = out(149:180,:,:,1:10); 
% right_PMv_pre = out(213:244,:,:,1:10); 
% left_M1_pre = out(245:276,:,:,1:10); 
% 
% out= cat(1,left_PMd_pre, left_PMv_pre, right_PMv_pre, left_M1_pre);

channel_electrode_map = struct;

PMv_left_pre = [173 175 177 179 169 171 166 168 170 172 174 176 178 180 154 160 162 164 157 159 161 163 165 167 149 151 153 155 150 152 158 156];
PMd_left_pre = [137 139 141 143 133 135 130 132 134 136 138 140 142 144 118 124 126 128 121 123 125 127 129 131 113 115 117 119 114 116 122 120];
PMd_right_pre = [];
PMv_right_pre = [220 218 216 214 219 217 215 213 231 229 227 225 223 221 228 226 224 222 244 242 240 238 236 234 232 230 235 233 243 241 239 237];
M1_left_pre = [270 272 274 276 269 271 273 275 257 259 261 263 265 267 262 264 266 268 246 248 254 252 250 256 258 260 253 255 245 247 249 251];
channel_electrode_map.pre_stroke = [PMv_left_pre,PMd_left_pre, PMd_right_pre, PMv_right_pre, M1_left_pre];

out_rearranged = out(channel_electrode_map.pre_stroke,:,:,:);
out = out_rearranged;
%% Save preprocessed data
% dataSorted = lfpData; 
cd(savepath)
% save(save_name, 'lfpData', '-v7.3') % output format: dataSorted = n_channels x time_samples x trials x blocks x sessions
save(save_name, 'out')





