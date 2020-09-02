cueOns_135 = zeros(1,n_trials);
cueOffs_135 = zeros(1,n_trials);
graspStarts_135 = zeros(1,n_trials);

fprintf('Loading 135 degree files (%d files)... \n', n_trials)
for f = 1:length(files_to_load_135)
    if f == 1
        load(files_to_load_135{f});
        lfpData_135 = F;
        channelFlags_135 = ChannelFlag;
        cueOns_135(f) = Events(1).times; cueOffs_135(f) = Events(5).times; graspStarts_135(f) = Events(9).times;
    else
        load(files_to_load_135{f});
        lfpData_135 = cat(3, lfpData_135, F);
        channelFlags_135 = cat(2, channelFlags_135, ChannelFlag);
        cueOns_135(f) = Events(1).times; cueOffs_135(f) = Events(5).times; graspStarts_135(f) = Events(9).times;
    end
    fprintf('File %d loaded... \n', f)
    
    if f == n_trials
        break
    end
end