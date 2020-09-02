function out = reshapeTrials(lfpData, lfpParam, taskParam, subsample_size, minimum_grasp_time, fs)
real_grasp_trial_inds = find(taskParam.timeEndGrasp - taskParam.timeStartGrasp > minimum_grasp_time);
real_grasp_trial_inds = real_grasp_trial_inds(1:end-1);

 
%timeToSamples(taskParam.timeTrialEnd, lfpParam.timeStartLFP, sampleRate)
grasp_durations = zeros(1,length(real_grasp_trial_inds));
n_trials_real_grasps = length(real_grasp_trial_inds);
for i = 1:n_trials_real_grasps
   startSampleInd = timeToSamples(taskParam.timeStartGrasp(real_grasp_trial_inds(i)),lfpParam.timeStartLFP, fs);
   endSampleInd = timeToSamples(taskParam.timeEndGrasp(real_grasp_trial_inds(i)),lfpParam.timeStartLFP, fs);
   grasp_data = lfpData(:, startSampleInd:endSampleInd);
   
   grasp_durations(i) = size(grasp_data,2); % just to get an idea of samples per grasp, could increase subsample_size!
   
   grasp_data_sample = grasp_data(:, fix(length(grasp_data)/2)-(subsample_size/2-1):fix(length(grasp_data)/2)+subsample_size/2);
   
   if i == 1
       out = grasp_data_sample;
   else
       out = cat(3, out, grasp_data_sample);
   end
   
end


end