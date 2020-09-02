%% Data path
save_name = 'dataSorted.mat';
datapath_pre = 'D:\ub_neuroComp\dancause_data\stroke\stroke_data\20180608Y';
datapath_post = 'D:\ub_neuroComp\dancause_data\stroke\stroke_data\post_stroke\20180710Y';
savepath = 'D:\ub_neuroComp\dancause_data\processing\4_preVSpost\left\export';
%% Add Paths
% restoredefaultpath
addpath(datapath_pre)
addpath(datapath_post)
addpath('D:\ub_neuroComp\dancause_data\processing\4_preVSpost\left')
addpath('D:\ub_neuroComp\dancause_data\processing\4_preVSpost\left\utils')
addpath(savepath)

datapaths = {datapath_pre, datapath_post};

excluded_channels = struct; excluded_channels.channelFlags = [];
channel_ids = struct; channel_ids.channels= [];
for cond = 1:2 %pre, post
    datapath = datapaths{cond};
    %% Detect and select files
    hand = {'Left'}; %'Left'
    precision_angle = {'Precision_45'}; %,'45','90','135'};
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
        inds = contains(file_names, hand) & contains(file_names, aligned_to) & contains(file_names, precision_angle) & ~contains(file_names, 'spikeFree');
    end

    files_to_load_right = file_names(inds);


    %% Load Data
    n_trials = 21;

    cueOns = zeros(1,n_trials);
    cueOffs = zeros(1,n_trials);
    graspStarts = zeros(1,n_trials);

    fprintf('Loading %d files... \n', n_trials)
    for f = 1:length(files_to_load_right)
        if f == 1
            load(files_to_load_right{f});
            lfpData_right = F;
            
            channelFlags_right = ChannelFlag;
            cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
        else
            load(files_to_load_right{f});
            lfpData_right = cat(3, lfpData_right, F);
            channelFlags_right = cat(2, channelFlags_right, ChannelFlag);
            cueOns(f) = Events(1).times; cueOffs(f) = Events(5).times; graspStarts(f) = Events(9).times;
        end
        fprintf('File %d loaded... \n', f)

        if f == n_trials
            break
        end
    end
    excluded_channels(cond).channelFlags = channelFlags_right;
    load([datapath,'\channel.mat'])
    channel_ids(cond).channels = Channel;
    %% Normalize
    if 1 
        for t = 1:size(lfpData_right,3)
            for ch = 1:size(lfpData_right,1)
                mu = mean(lfpData_right(ch,:,t));
                sdev = std(lfpData_right(ch,:,t));
                lfpData_right(ch,:,t) = (lfpData_right(ch,:,t) - mu) / (sdev+0.00001);
            end
        end
        fprintf('Data normalized...\n')
    end
    %% Seperate actions
    sample_duration = 0.20;

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
    if cond == 1
        out = cat(4, baselines, pre_grasps, reaches, grasps, post_grasps);
    else
        out = cat(4, out, baselines, pre_grasps, reaches, grasps, post_grasps);
    end
    clear baselines
    clear pre_grasps
    clear reaches
    clear grasps
    clear post_grasps

end
%% Rearrange channels (hardcoded)
ch_names = cell(size(channel_ids(1).channels,2),size(channel_ids,2));
ch_groups = cell(size(channel_ids(1).channels,2),size(channel_ids,2));
for cond = 1:size(channel_ids,2)
    for ch = 1:size(channel_ids(1).channels,2)
        ch_names{ch,cond}=channel_ids(cond).channels(ch).Name;
        ch_groups{ch,cond}=channel_ids(cond).channels(ch).Group;
    end
end

left_PMd_pre = out(113:144,:,:,1:5); left_PMd_post = out(17:48,:,:,6:10); 
left_PMv_pre = out(149:180,:,:,1:5); left_PMv_post = out(113:144,:,:,6:10);
right_PMv_pre = out(213:244,:,:,1:5); right_PMv_post = out(245:276,:,:,6:10);
left_M1_pre = out(245:276,:,:,1:5); left_M1_post = out(213:244,:,:,6:10);

out_rearranged_pre = cat(1,left_PMd_pre, left_PMv_pre, right_PMv_pre, left_M1_pre);
out_rearranged_post= cat(1,left_PMd_post, left_PMv_post, right_PMv_post, left_M1_post);
out = cat(4, out_rearranged_pre, out_rearranged_post);

%% Remove excluded_channels
% assume same channels removed across trials within a session
if 0 
    for cond = 1:size(excluded_channels,2)
    bad_chans = find(excluded_channels(cond).channelFlags(:,1) == -1);

    fprintf('%d bad channels in condition %d...\n',length(bad_chans),cond)
    end
end
%% Save preprocessed data
% dataSorted = lfpData; 
cd(savepath)
% save(save_name, 'lfpData', '-v7.3') % output format: dataSorted = n_channels x time_samples x trials x blocks x sessions
save(save_name, 'out')





