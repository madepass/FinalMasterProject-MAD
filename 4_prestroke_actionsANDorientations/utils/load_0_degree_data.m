cueOns_0 = zeros(1,n_trials);
cueOffs_0 = zeros(1,n_trials);
graspStarts_0 = zeros(1,n_trials);

fprintf('Loading 0 degree data (%d files)... \n', n_trials)
for f = 1:length(files_to_load_0)
    if f == 1
        load(files_to_load_0{f});
        lfpData_0 = F;
        channelFlags_0 = ChannelFlag;
        cueOns_0(f) = Events(1).times; cueOffs_0(f) = Events(5).times; graspStarts_0(f) = Events(9).times;
    else
        load(files_to_load_0{f});
        lfpData_0 = cat(3, lfpData_0, F);
        channelFlags_0 = cat(2, channelFlags_0, ChannelFlag);
        cueOns_0(f) = Events(1).times; cueOffs_0(f) = Events(5).times; graspStarts_0(f) = Events(9).times;
    end
    fprintf('File %d loaded... \n', f)
    
    if f == n_trials
        break
    end
end