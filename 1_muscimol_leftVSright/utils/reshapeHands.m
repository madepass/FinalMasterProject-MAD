function out = reshapeHands(out, taskParam, minimum_grasp_time)
real_grasp_trial_inds = find(taskParam.timeEndGrasp - taskParam.timeStartGrasp > minimum_grasp_time);
real_grasp_trial_inds = real_grasp_trial_inds(1:end-1);

left_trials = taskParam.condCode(real_grasp_trial_inds) == 1 | taskParam.condCode(real_grasp_trial_inds) == 3;
right_trials = taskParam.condCode(real_grasp_trial_inds) == 2 | taskParam.condCode(real_grasp_trial_inds) == 4;

aux1 = out(:,:,left_trials);
aux1 = aux1(:,:,1:min(sum(left_trials), sum(right_trials)));
aux2 = out(:,:,right_trials);
aux2 = aux2(:,:,1:min(sum(left_trials), sum(right_trials)));

out = cat(4, aux1, aux2);
end