## Import libraries
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

## Set datapath
data_path = 'D:\\ub_neuroComp\\dancause_data\\processing\\3_prestroke_actionsANDorientations\\export\\subNS1\\'

## Import & aggregate
perf_files = [file_name for file_name in os.listdir(data_path) if 'perf_' in file_name and 'shuf' not in file_name]
perf_shuff_files = [file_name for file_name in os.listdir(data_path) if 'perf_' in file_name and 'shuf' in file_name]
# for reference: perf = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
for f, (file_name, file_name_shuff) in enumerate(zip(perf_files,perf_shuff_files)):
    if f == 0: 
        perf = np.load(data_path + file_name)[f,0,:,:]
        perf_shuff = np.load(data_path + file_name_shuff)[f,0,:,:]
    else: 
        perf = np.dstack((perf,np.load(data_path + file_name)[f,0,:,:])) # (n_reps, log/1-NN, bands)
        perf_shuff = np.dstack((perf_shuff,np.load(data_path + file_name_shuff)[f,0,:,:])) # (n_reps, log/1-NN, bands)
        
## Plot Accuracy
plt.figure(figsize=[4,3])
plt.errorbar(np.arange(perf.shape[2]),perf[:,0,:].mean(axis=0), perf[:,0,:].std(axis=0)/perf.shape[0], label = 'MLR', color='blue')
plt.errorbar(np.arange(perf.shape[2]),perf[:,1,:].mean(axis=0), perf[:,1,:].std(axis=0)/perf.shape[0], label = '1-NN', color='orange')
plt.errorbar(np.arange(perf.shape[2]),perf_shuff[:,0,:].mean(axis=0), perf_shuff[:,0,:].std(axis=0)/perf.shape[0], label = 'MLR-surrogate', color='blue', linestyle='dotted')
plt.errorbar(np.arange(perf.shape[2]),perf_shuff[:,1,:].mean(axis=0), perf_shuff[:,1,:].std(axis=0)/perf.shape[0], label = '1-NN-surrogate', color='orange', linestyle='dotted')
plt.ylabel('Mean Accuracy')
plt.xlabel('Frequency Band')
plt.title('Accuracy vs. Frequency Band')
plt.legend(loc='best',frameon=False, fontsize=7)
plt.grid(alpha=.4)
sns.despine()
plt.tight_layout()
plt.savefig(data_path + 'accuracy_lineplot.png')
print('Printed accuracy lineplot...')


if 1: 
    ## Import & aggregate
    confmat_files = [file_name for file_name in os.listdir(data_path) if 'conf_' in file_name and 'png' not in file_name]
    confusion_matrix = []
    for f, file_name in enumerate(confmat_files):
            confusion_matrix.append(np.load(data_path + file_name)[f,0,:,:,:])
    confusion_matrix = np.stack(confusion_matrix, axis = 0)

    ## Compute session confusion
    session_confusion_mlr = []; 
    session_confusion_1nn = []; 
    for band in range(confusion_matrix.shape[0]):
        mean_confmat_mlr = np.mean(confusion_matrix[band,:,0,:,:], axis = 0)
        session_confusion_mlr.append((mean_confmat_mlr[5:,0:5].sum() + mean_confmat_mlr[0:5,5:].sum())/np.sum(mean_confmat_mlr))
        mean_confmat_1nn = np.mean(confusion_matrix[band,:,1,:,:], axis = 0)
        session_confusion_1nn.append((mean_confmat_1nn[5:,0:5].sum() + mean_confmat_1nn[0:5,5:].sum())/np.sum(mean_confmat_1nn))

    ## Plot session confusion
    plt.figure(figsize=[4,3])
    plt.plot(np.arange(confusion_matrix.shape[0]), session_confusion_mlr, label = 'MLR', color = 'blue')
    plt.plot(np.arange(confusion_matrix.shape[0]), session_confusion_1nn, label = '1-NN', color = 'orange')
    plt.ylabel('Mean session confusion')
    plt.xlabel('Frequency Band')
    plt.title('Session Confusion vs. Frequency Band')
    plt.legend(loc='best',frameon=False, fontsize=7)
    plt.grid(alpha=.4)
    sns.despine()
    plt.tight_layout()
    plt.savefig(data_path + 'session_confusion_lineplot.png')
    print('Printed session confusion lineplot...')
