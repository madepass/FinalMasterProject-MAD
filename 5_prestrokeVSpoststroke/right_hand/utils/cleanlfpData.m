function lfpData = cleanlfpData(lfpData, cleaningSettings, fs)


if cleaningSettings.switches(1)
    lower_cutoff = cleaningSettings.bandpass(1);
    upper_cutoff = cleaningSettings.bandpass(2);
    lfpData = bandpass(lfpData, [lower_cutoff, upper_cutoff], fs); % bandpass
end

if cleaningSettings.switches(2)
    lfpData = lfpData - mean(lfpData,2); % de-mean
end

if cleaningSettings.switches(3) % remove outliers
    n_stds = cleaningSettings.n_stds;
    outlier_lims = cat(2,mean(lfpData,2) - n_stds * std(lfpData,0,2),mean(lfpData,2) + n_stds * std(lfpData,0,2)); % remove outliers
    for i = 1:size(lfpData,1)
       lfpData(i,lfpData(i,:) < outlier_lims(i,1)) = nan;
       lfpData(i,lfpData(i,:) > outlier_lims(i,2)) = nan;
    end
end

if cleaningSettings.switches(4) % interpolate nans
    for i = 1:size(lfpData,1)
       for j = 1:size(lfpData,2)
            if isnan(lfpData(i,j))
               if j < 11
                    lfpData(i,j) = nanmean(lfpData(i,j: j + 10));
               elseif j > size(lfpData,2) - 10
                    lfpData(i,j) = nanmean(lfpData(i,j - 10 : j)); 
               else
                   lfpData(i,j) = nanmean(lfpData(i,j - 10 : j + 10));
               end
            end
       end
    end
end

end