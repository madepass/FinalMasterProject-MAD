fprintf('Extracting baselines...\n')
for t = 1: n_trials %baselines
    min_ind = ceil(median(find(Time < cueOns_0(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time < cueOns_0(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        baselines_0 = lfpData_0(:,min_ind:max_ind,t);
    else
        baseline_0 = lfpData_0(:,min_ind:max_ind,t);
        baselines_0 = cat(3,baselines_0,baseline_0);
    end   
end

fprintf('Extracting pre_grasps...\n')
for t = 1: n_trials %pre_grasps
    min_ind = ceil(median(find(Time > cueOns_0(t) & Time < cueOffs_0(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOns_0(t) & Time < cueOffs_0(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        pre_grasps_0 = lfpData_0(:,min_ind:max_ind,t);
    else
        pre_grasp_0 = lfpData_0(:,min_ind:max_ind,t);
        pre_grasps_0 = cat(3,pre_grasps_0,pre_grasp_0);
    end   
end
fprintf('Extracting reaches...\n')
for t = 1: n_trials %reaches
    min_ind = ceil(median(find(Time > cueOffs_0(t) & Time < graspStarts_0(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOffs_0(t) & Time < graspStarts_0(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        reaches_0 = lfpData_0(:,min_ind:max_ind,t);
    else
        reach_0 = lfpData_0(:,min_ind:max_ind,t);
        reaches_0 = cat(3,reaches_0,reach_0);
    end   
end
fprintf('Extraction grasps...\n')
for t = 1: n_trials %grasps
    min_ind = ceil(median(find(Time > graspStarts_0(t) & Time < (graspStarts_0(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts_0(t) & Time < (graspStarts_0(t)+sample_duration)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        grasps_0 = lfpData_0(:,min_ind:max_ind,t);
    else
        grasp_0 = lfpData_0(:,min_ind:max_ind,t);
        grasps_0 = cat(3,grasps_0,grasp_0);
    end   
end
fprintf('Extracting post_grasps...\n')
for t = 1: n_trials %post_grasps
    min_ind = ceil(median(find(Time > (graspStarts_0(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts_0(t)+sample_duration))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        post_grasps_0 = lfpData_0(:,min_ind:max_ind,t);
    else
        post_grasp_0 = lfpData_0(:,min_ind:max_ind,t);
        post_grasps_0 = cat(3,post_grasps_0,post_grasp_0);
    end   
end