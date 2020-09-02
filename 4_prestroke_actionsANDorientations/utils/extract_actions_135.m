fprintf('Extracting baselines...\n')
for t = 1: n_trials %baselines
    min_ind = ceil(median(find(Time < cueOns_135(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time < cueOns_135(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        baselines_135 = lfpData_135(:,min_ind:max_ind,t);
    else
        baseline_135 = lfpData_135(:,min_ind:max_ind,t);
        baselines_135 = cat(3,baselines_135,baseline_135);
    end   
end

fprintf('Extracting pre_grasps...\n')
for t = 1: n_trials %pre_grasps
    min_ind = ceil(median(find(Time > cueOns_135(t) & Time < cueOffs_135(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOns_135(t) & Time < cueOffs_135(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        pre_grasps_135 = lfpData_135(:,min_ind:max_ind,t);
    else
        pre_grasp_135 = lfpData_135(:,min_ind:max_ind,t);
        pre_grasps_135 = cat(3,pre_grasps_135,pre_grasp_135);
    end   
end
fprintf('Extracting reaches...\n')
for t = 1: n_trials %reaches
    min_ind = ceil(median(find(Time > cueOffs_135(t) & Time < graspStarts_135(t)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > cueOffs_135(t) & Time < graspStarts_135(t)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        reaches_135 = lfpData_135(:,min_ind:max_ind,t);
    else
        reach_135 = lfpData_135(:,min_ind:max_ind,t);
        reaches_135 = cat(3,reaches_135,reach_135);
    end   
end
fprintf('Extraction grasps...\n')
for t = 1: n_trials %grasps
    min_ind = ceil(median(find(Time > graspStarts_135(t) & Time < (graspStarts_135(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts_135(t) & Time < (graspStarts_135(t)+sample_duration)))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        grasps_135 = lfpData_135(:,min_ind:max_ind,t);
    else
        grasp_135 = lfpData_135(:,min_ind:max_ind,t);
        grasps_135 = cat(3,grasps_135,grasp_135);
    end   
end
fprintf('Extracting post_grasps...\n')
for t = 1: n_trials %post_grasps
    min_ind = ceil(median(find(Time > (graspStarts_0(t)+sample_duration)))) - floor(samples_per_sample / 2);
    max_ind = ceil(median(find(Time > graspStarts_0(t)+sample_duration))) + floor(samples_per_sample / 2) - 1;
    if t == 1
        post_grasps_135 = lfpData_135(:,min_ind:max_ind,t);
    else
        post_grasp_135 = lfpData_135(:,min_ind:max_ind,t);
        post_grasps_135 = cat(3,post_grasps_135,post_grasp_135);
    end   
end