%% Data path
save_name = 'dataSorted.mat';
datapath = 'D:\ub_neuroComp\dancause_data\stroke\stroke_data\post_stroke\20180710Y';
savepath = 'D:\ub_neuroComp\dancause_data\processing\2b_poststroke_actions\export';
%% Add Paths
% restoredefaultpath
addpath(datapath)
addpath('D:\ub_neuroComp\dancause_data\processing\2b_poststroke_actions')
addpath('D:\ub_neuroComp\dancause_data\processing\2b_poststroke_actions\utils')
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

files_to_load_right = file_names(inds);

% hand = {'Left'}; % , 'Left'
% precision_angle = {'Precision_0'};%,'45','90','135'};
% aligned_to = {'GraspStart'}; %'CueOn'
% spikes = {'spikeFree'}; % 
% % data_options = [hand, aligned_to, spikes];
% 
% all_files = dir(datapath);
% file_names = cell(1,length(all_files));
% for i = 1:length(file_names)
%    file_names{i} = all_files(i).name; 
% end
% 
% if strcmp(spikes{1}, 'spikeFree')
%     inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle) & contains(file_names, spikes); 
% else
%     inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle);
% end
% 
% files_to_load_left = file_names(inds);
%% Load Data
% load('2014_10_14_G_Pre.mat') % already exported
% load("2014_10_14_G_T00.mat");
n_trials = 21;

cueOns = zeros(1,n_trials);
cueOffs = zeros(1,n_trials);
graspStarts = zeros(1,n_trials);
% fprintf('Loading Left %d files... \n', length(files_to_load_left))
% for f = 1:length(files_to_load_left)
%     if f == 1
%         load(files_to_load_left{f});
%         lfpData_left = F;
%         %channelFlags_left = ChannelFlag;
%         cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
%     else
%         load(files_to_load_left{f});
%         lfpData_left = cat(3, lfpData_left, F);
%         %channelFlags_left = cat(2, channelFlags_left, ChannelFlag);
%         cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
%     end
%     fprintf('File %d loaded... \n', f)
%     
%     if f == n_trials
%         break
%     end
% end

fprintf('Loading Right %d files... \n', length(files_to_load_right))
for f = 1:length(files_to_load_right)
    if f == 1
        load(files_to_load_right{f});
        lfpData_right = F;
        channelFlags = ChannelFlag;
        cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
    else
        load(files_to_load_right{f});
        lfpData_right = cat(3, lfpData_right, F);
        channelFlags = cat(2, channelFlags, ChannelFlag);
        cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
    end
    fprintf('File %d loaded... \n', f)
    
    if f == n_trials
        break
    end
end

load([datapath,'\channel.mat'])
channel_ids(1).channels = Channel;
%% Normalize
for t = 1:size(lfpData_right,3)
        for ch = 1:size(lfpData_right,1)
            mu = mean(lfpData_right(ch,:,t));
            sdev = std(lfpData_right(ch,:,t));
            lfpData_right(ch,:,t) = (lfpData_right(ch,:,t) - mu) / (sdev+0.00001);
        end
end
    fprintf('Data normalized...\n')
%% Seperate actions
sample_duration = 0.24;

min_baseline = min(cueOns+Time(end));
min_pre_grasp = min(cueOffs - cueOns);
min_reach = min(graspStarts - cueOffs);
% min_grasp % grasp set to graspStart + sample_duration 
min_post_grasp = min(Time(end) - (graspStarts + sample_duration)); % check for too large duration
if  min_baseline < sample_duration ||  min_pre_grasp < sample_duration || min_reach < sample_duration ||  min_post_grasp < sample_duration
    error('Sample duration too large. Cannot make samples with identical durations.')
end
trial_duration = Time(end) - Time(1);
fs = length(Time)  / trial_duration;
samples_per_sample = floor(sample_duration * fs);

fprintf('Extracting baselines...\n')
for t = 1: n_trials %baselines
    min_ind = ceil(median(find(Time < cueOns(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time < cueOns(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        baselines = lfpData_right(:,min_ind:max_ind,t);
    else
        baseline = lfpData_right(:,min_ind:max_ind,t);
        baselines = cat(3,baselines,baseline);
    end   
end

fprintf('Extracting pre_grasps...\n')
for t = 1: n_trials %pre_grasps
    min_ind = ceil(median(find(Time > cueOns(t) & Time < cueOffs(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOns(t) & Time < cueOffs(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        pre_grasps = lfpData_right(:,min_ind:max_ind,t);
    else
        pre_grasp = lfpData_right(:,min_ind:max_ind,t);
        pre_grasps = cat(3,pre_grasps,pre_grasp);
    end   
end
fprintf('Extracting reaches...\n')
for t = 1: n_trials %reaches
    min_ind = ceil(median(find(Time > cueOffs(t) & Time < graspStarts(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOffs(t) & Time < graspStarts(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        reaches = lfpData_right(:,min_ind:max_ind,t);
    else
        reach = lfpData_right(:,min_ind:max_ind,t);
        reaches = cat(3,reaches,reach);
    end   
end
fprintf('Extraction grasps...\n')
for t = 1: n_trials %grasps
    min_ind = ceil(median(find(Time > graspStarts(t) & Time < (graspStarts(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts(t) & Time < (graspStarts(t)+sample_duration)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        grasps = lfpData_right(:,min_ind:max_ind,t);
    else
        grasp = lfpData_right(:,min_ind:max_ind,t);
        grasps = cat(3,grasps,grasp);
    end   
end
fprintf('Extracting post_grasps...\n')
for t = 1: n_trials %post_grasps
    min_ind = ceil(median(find(Time > (graspStarts(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts(t)+sample_duration))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        post_grasps = lfpData_right(:,min_ind:max_ind,t);
    else
        post_grasp = lfpData_right(:,min_ind:max_ind,t);
        post_grasps = cat(3,post_grasps,post_grasp);
    end   
end

%% Join actions
out = cat(4, baselines, pre_grasps, reaches, grasps, post_grasps);
clear baselines
clear pre_grasps
clear reaches
clear grasps
clear post_grasps
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
% left_PMd_pre = out(113:144,:,:,:); 
% left_PMv_pre = out(149:180,:,:,:); 
% right_PMv_pre = out(213:244,:,:,:); 
% left_M1_pre = out(245:276,:,:,:); 
% 
% out= cat(1,left_PMd_pre, left_PMv_pre, right_PMv_pre, left_M1_pre);
channel_electrode_map = struct;

PMv_left_post = [137 139 141 143 133 135 130 132 134 136 138 140 142 144 118 124 126 128 121 123 125 127 129 131 113 115 117 119 114 116 122 120];
PMd_left_post = [37 44 46 48 22 24 26 28 30 32 34 36 39 41 29 35 38 40 42 43 45 47 18 20 17 19 21 23 25 27 33 31];
PMd_right_post = [];
PMv_right_post = [270 272 274 276 269 271 273 275 257 259 261 263 265 267 262 264 266 268 246 248 254 252 250 256 258 260 253 255 245 247 249 251];
M1_left_post = [230 232 235 237 233 240 242 244 241 243 214 216 218 220 222 224 226 228 221 223 229 227 225 231 234 236 238 239 213 215 217 219];
channel_electrode_map.post_stroke = [PMv_left_post,PMd_left_post, PMd_right_post, PMv_right_post, M1_left_post];

out_rearranged = out(channel_electrode_map.post_stroke,:,:,:);
out = out_rearranged;

%% Save preprocessed data
% dataSorted = lfpData; 
cd(savepath)
% save(save_name, 'lfpData', '-v7.3') % output format: dataSorted = n_channels x time_samples x trials x blocks x sessions
save(save_name, 'out')





