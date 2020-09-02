# greek_band = 8; classification_feature = 0; i_sub = 1;
#%% Import Sys vars
import sys 
greek_band = int(sys.argv[1]) - 1 # 1-9, delta theta alpha beta etc. 
classification_feature = int(sys.argv[2]) - 1# 1-3, pow, cov, corr
# sys.exit("manually exited program")

#%%  Import libraries
import os
import numpy as np
import scipy.signal as spsg
import scipy.stats as stt
import scipy.io as sio
import sklearn.linear_model as skllm
import sklearn.neighbors as sklnn
import sklearn.preprocessing as skprp
import sklearn.pipeline as skppl
import sklearn.feature_selection as skfs
import sklearn.model_selection as skms
import sklearn.metrics as skm
import matplotlib.pyplot as pp
import networkx as nx
from sklearn.metrics import roc_curve, roc_auc_score, plot_roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
import pandas as pd      
import seaborn as sns
from copy import deepcopy
#%% 
os.chdir('D:\\ub_neuroComp\\dancause_data\\processing\\2_prestroke_actions\\export')
#%% general info

subjects = [1]
mat_names = ['','dataSorted']

for i_sub in subjects:

    res_dir = 'subNS'+str(i_sub)+'/'
    if not os.path.exists(res_dir):
        print('create directory:',res_dir)
        os.makedirs(res_dir)
    
    # 
    cmapcolours = ['Blues','Greens','Oranges','Blues','Greens','Oranges','Blues','Greens','Oranges' ]
    listcolours = ['b','g','r','b','g','r','b','g','r']
    
    measure_labels = ['pow','cov','corr']
    n_measures = len(measure_labels)
    
    freq_bands = ['theta', 'alpha','beta','low gamma', 'high gamma', 'low ripple', 'high ripple', 'low multi-unit', 'high multi-unit']
    n_bands = len(freq_bands)

    #%% load data
    
    ts_tmp = sio.loadmat(mat_names[i_sub])['out'] # [N,T,n_trials,motiv]   
    n_motiv = ts_tmp.shape[-1] #mad
    N = ts_tmp.shape[0] #mad
    n_trials = ts_tmp.shape[2] #mad
    T = ts_tmp.shape[1] #mad
    
    # discard silent channels
    invalid_ch = np.logical_or(np.abs(ts_tmp[:,:,0,0]).max(axis=1)==0, np.isnan(ts_tmp[:,0,0,0]))
    valid_ch = np.logical_not(invalid_ch)
    ts_tmp = ts_tmp[valid_ch,:,:,:]
    N = valid_ch.sum()
    
    # get time series for each block
    ts = np.zeros([n_motiv,n_trials,T,N])
    for i_motiv in range(n_motiv):
        for i_trial in range(n_trials):
            # swap axes for time and channels
            ts[i_motiv,i_trial,:,:] = ts_tmp[:,:,i_trial,i_motiv].T
    
    del ts_tmp # clean memory
    
    mask_tri = np.tri(N,N,-1,dtype=np.bool) # mask to extract lower triangle of matrix
    
    
    #%% get channel positions for plot
    
    # node positions for circular layout with origin at bottom
    if 1:
        var_dict = np.genfromtxt('coordinates.sfp')
        
        x_sensor = var_dict[:,1]
        y_sensor = var_dict[:,2]
        
        print(x_sensor)
        print(y_sensor)
        
            
        # positions of sensors
        pos_circ = dict()
        for i in range(N):
            pos_circ[i] = np.array([x_sensor[i], y_sensor[i]])
            
    # channel labels
    ch_labels = dict()
    for i in range(N):
        ch_labels[i] = i+1
    
    # matrices to retrieve input/output channels from connections in support network
    row_ind = np.repeat(np.arange(N).reshape([N,-1]),N,axis=1)
    col_ind = np.repeat(np.arange(N).reshape([-1,N]),N,axis=0)    
    row_ind = row_ind[mask_tri]
    col_ind = col_ind[mask_tri]
    
    
    #%% classifier and learning parameters
    
    # MLR adapted for recursive feature elimination (RFE)
    class RFE_pipeline(skppl.Pipeline):
        def fit(self, X, y=None, **fit_params):
            """simply extends the pipeline to recover the coefficients (used by RFE) from the last element (the classifier)
            """
            super(RFE_pipeline, self).fit(X, y, **fit_params)
            self.coef_ = self.steps[-1][-1].coef_
            return self
    
    c_MLR = RFE_pipeline([('std_scal',skprp.StandardScaler()),('clf',skllm.LogisticRegression(C=10, penalty='l2', multi_class='multinomial', solver='lbfgs', max_iter=5000))])
    
    # nearest neighbor
    c_1NN = sklnn.KNeighborsClassifier(n_neighbors=1, algorithm='brute', metric='correlation')
     
    # cross-validation scheme
    cv_schem = skms.StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    n_rep = 10 # number of repetitions
    
    # RFE wrappers
    RFE_pow = skfs.RFE(c_MLR, n_features_to_select=3)
    RFE_FC = skfs.RFE(c_MLR, n_features_to_select=10)
    
    # record classification performance 
    perf = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
    perf_shuf = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
    conf_matrix = np.zeros([n_bands,n_measures,n_rep,2,n_motiv,n_motiv]) # (fourthindex: MLR/1NN)
    rk_pow = np.zeros([n_bands,n_rep,N],dtype=np.int) # RFE rankings for power (N feature)
    rk_FC = np.zeros([n_bands,2,n_rep,int(N*(N-1)/2)],dtype=np.int) # RFE rankings for FC-type measures (N(N-1)/2 feature)
    pearson_corr_rk = np.zeros([n_bands,n_measures,int(n_rep*(n_rep-1)/2)]) # stability of rankings measured by Pearson correlation
    
    #MLR_probs = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
    #KNN_probs = np.zeros([n_bands,n_measures,n_rep,2]) # (last index: MLR/1NN)
    
    #%% loop over the measures and frequency bands
    
    # for i_band in range(n_bands):
    i_band = greek_band #################################################################

    freq_band = freq_bands[i_band]
    
    # band-pass filtering (alpha, beta, gamma)
    n_order = 3
    sampling_freq = 2034.5 # sampling rate
    
    if freq_band=='theta':
        low_f = 4./sampling_freq
        high_f = 7./sampling_freq
    elif freq_band=='alpha':    
        # beta
        low_f = 8./sampling_freq
        high_f = 15./sampling_freq
    elif freq_band=='beta':
        # gamma
        low_f = 15./sampling_freq
        high_f = 30./sampling_freq
    elif freq_band=='low gamma':
        # gamma
        low_f = 30./sampling_freq
        high_f = 70./sampling_freq
    elif freq_band=='high gamma':
        # gamma
        low_f = 70./sampling_freq
        high_f = 100./sampling_freq
    elif freq_band=='low ripple':
        # gamma
        low_f = 100./sampling_freq
        high_f = 150./sampling_freq
    elif freq_band=='high ripple':
        # gamma
        low_f = 150./sampling_freq
        high_f = 200./sampling_freq
    elif freq_band=='low multi-unit':
        # gamma
        low_f = 200./sampling_freq
        high_f = 500./sampling_freq
    elif freq_band=='high multi-unit':
        # gamma
        low_f = 500./sampling_freq
        high_f = 1000./sampling_freq
    else:
        raise NameError('unknown filter')
    
    # apply filter
    b,a = spsg.iirfilter(n_order, [low_f,high_f], btype='bandpass', ftype='butter')
    filtered_ts = spsg.filtfilt(b, a, ts, axis=2)
        
        # for i_measure in range(n_measures):
    i_measure = classification_feature #####################################################################################
            
    print('frequency band, measure:', freq_band, measure_labels[i_measure])
        
            # need safe margins to remove filtering effect? seems like no
    if False:
        pp.plot(filtered_ts[0,0,:,0])

    if i_measure == 0: # power of signal within each sliding window (rectification by absolute value)
        # create the design matrix [samples,features]
        vect_features = np.abs(filtered_ts).mean(axis=2)
        
    else: # covariance or correlation
        EEG_FC = np.zeros([n_motiv,n_trials,N,N]) # dynamic FC = covariance or Pearson correlation of signal within each sliding window
        for i_motiv in range(n_motiv):
            for i_trial in range(n_trials):
                ts_tmp = filtered_ts[i_motiv,i_trial,:,:]
                ts_tmp -= np.outer(np.ones(T),ts_tmp.mean(0))
                EEG_FC[i_motiv,i_trial,:,:] = np.tensordot(ts_tmp,ts_tmp,axes=(0,0)) / float(T-1)
                if i_measure==2: # correlation, not covariance
                    EEG_FC[i_motiv,i_trial,:,:] /= np.sqrt(np.outer(EEG_FC[i_motiv,i_trial,:,:].diagonal(),EEG_FC[i_motiv,i_trial,:,:].diagonal()))

        # vectorize the connectivity matrices to obtain the design matrix [samples,features]
        vect_features = EEG_FC[:,:,mask_tri]
                
    # labels of sessions for classification (train+test)
    labels = np.zeros([n_motiv,n_trials], dtype=np.int) # 0 = M0, 1 = M1, 2 = M2
    for i in range(labels.shape[0]):
        labels[i,:] = i
    
    # vectorize dimensions motivation levels and trials
    mask_motiv_trials = np.ones([n_motiv,n_trials], dtype=np.bool)
    vect_features = vect_features[mask_motiv_trials,:]
    labels = labels[mask_motiv_trials]
            
    # MAD check vect_features for nans
    contain_nan = []
    for row in vect_features:
       # med = np.median(row)
        if sum(np.isnan(row)) > 0:
            contain_nan.append(1)
        else:
            contain_nan.append(0)
    if sum(contain_nan) > 0: print(f'{sum(contain_nan)} rows containing nans detected in vect_features')
    
    for row, has_nan in enumerate(contain_nan):
        if has_nan:
           vect_features[row, np.isnan(vect_features[row,:])] = np.nanmedian(vect_features[row,:])
    if sum(contain_nan) > 0: print('Nans replaced by median of trial')
    

    ################
    # repeat classification for several splits for indices of sliding windows (train/test sets)
    for i_rep in range(n_rep):
        for ind_train, ind_test in cv_schem.split(vect_features,labels): # false loop, just 1 
            # train and test for original data
            c_MLR.fit(vect_features[ind_train,:], labels[ind_train]); 
            if i_rep == 0: 
                c_MLR_original = deepcopy(c_MLR)
                ind_train_original = deepcopy(ind_train)
                ind_test_original = deepcopy(ind_test) 
            perf[i_band,i_measure,i_rep,0] = c_MLR.score(vect_features[ind_test,:], labels[ind_test])
            conf_matrix[i_band,i_measure,i_rep,0,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_MLR.predict(vect_features[ind_test,:]))  
            
            c_1NN.fit(vect_features[ind_train,:], labels[ind_train])
            perf[i_band,i_measure,i_rep,1] = c_1NN.score(vect_features[ind_test,:], labels[ind_test])
            conf_matrix[i_band,i_measure,i_rep,1,:,:] += skm.confusion_matrix(y_true=labels[ind_test], y_pred=c_1NN.predict(vect_features[ind_test,:]))  
 
            
            # shuffled performance distributions
            shuf_labels = np.random.permutation(labels)
    
            c_MLR.fit(vect_features[ind_train,:], shuf_labels[ind_train])
            perf_shuf[i_band,i_measure,i_rep,0] = c_MLR.score(vect_features[ind_test,:], shuf_labels[ind_test])
     
            c_1NN.fit(vect_features[ind_train,:], shuf_labels[ind_train])
            perf_shuf[i_band,i_measure,i_rep,1] = c_1NN.score(vect_features[ind_test,:], shuf_labels[ind_test])
            
            # RFE for MLR
            if i_measure == 0: # power
                RFE_pow.fit(vect_features[ind_train,:], labels[ind_train])
                rk_pow[i_band,i_rep,:] = RFE_pow.ranking_
            else: # covariance or correlation
                RFE_FC.fit(vect_features[ind_train,:], labels[ind_train])
                rk_FC[i_band,i_measure-1,i_rep,:] = RFE_FC.ranking_
    
    if 1:
        # check stability RFE rankings
        # for i_band in range(n_bands):
            # for i_measure in range(n_measures): ##########################
                i_cnt = 0
                for i_rep1 in range(n_rep):
                    for i_rep2 in range(i_rep1):
                        pearson_corr_rk[i_band,0,i_cnt] = stt.pearsonr(rk_pow[i_band,i_rep1,:],rk_pow[i_band,i_rep2,:])[0]
                        pearson_corr_rk[i_band,1,i_cnt] = stt.pearsonr(rk_FC[i_band,0,i_rep1,:],rk_FC[i_band,0,i_rep2,:])[0]
                        pearson_corr_rk[i_band,2,i_cnt] = stt.pearsonr(rk_FC[i_band,1,i_rep1,:],rk_FC[i_band,1,i_rep2,:])[0]
                        i_cnt += 1
        
    # save results       
    np.save(res_dir+'perf_'+str(classification_feature+1)+str(greek_band+1)+'.npy',perf)
    np.save(res_dir+'perf_shuf_'+str(classification_feature+1)+str(greek_band+1)+'.npy',perf_shuf)
    np.save(res_dir+'conf_matrix_'+str(classification_feature+1)+str(greek_band+1)+'.npy',conf_matrix)
    np.save(res_dir+'rk_pow_'+str(classification_feature+1)+str(greek_band+1)+'.npy',rk_pow)
    np.save(res_dir+'rk_FC_'+str(classification_feature+1)+str(greek_band+1)+'.npy',rk_FC)
    np.save(res_dir+'pearson_corr_rk_'+str(classification_feature+1)+str(greek_band+1)+'.npy',pearson_corr_rk)
    
    
    #%% plots
    fmt_grph = 'png'
    
    
    # for i_band in range(n_bands):
    freq_band = freq_bands[i_band]
      #  for i_measure in range(n_measures):
    measure_label = measure_labels[i_measure]
    
    # the chance level is defined as the trivial classifier that predicts the label with more occurrences 
    chance_level = np.max(np.unique(labels, return_counts=True)[1]) / labels.size
    
    # plot performance and surrogate
    pp.figure(figsize=[4,3])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.violinplot(perf[i_band,i_measure,:,0],positions=[-0.2],widths=[0.3])
    pp.violinplot(perf[i_band,i_measure,:,1],positions=[0.2],widths=[0.3])
    pp.violinplot(perf_shuf[i_band,i_measure,:,0],positions=[0.8],widths=[0.3])
    pp.violinplot(perf_shuf[i_band,i_measure,:,1],positions=[1.2],widths=[0.3])
    pp.plot([-1,2],[chance_level]*2,'--k')
    pp.axis(xmin=-0.6,xmax=1.6,ymin=0,ymax=1.05)
    pp.xticks([0,1],['Correct Labels','Shuffled Labels'],fontsize=8)
    pp.ylabel('Accuracy',fontsize=8)
    pp.title(freq_band+', '+measure_label)
    pp.tight_layout()
    #pp.savefig(res_dir+f'accuracy_{classification_feature+1}{greek_band+1}_'+freq_band+'_'+measure_label + '.' + fmt_grph,bbox_inches='tight')
    pp.savefig(res_dir+f'accuracy_{classification_feature+1}{greek_band+1}_'+freq_band+'_'+measure_label + '.' + fmt_grph)
    pp.close()

    # plot confusion matrix for MLR
    pp.figure(figsize=[4,3])
    pp.axes([0.2,0.2,0.7,0.7])
    pp.imshow(conf_matrix[i_band,i_measure,:,0,:,:].mean(0), vmin=0, cmap=cmapcolours[i_band])
    pp.colorbar()
    pp.xlabel('True Label',fontsize=8)
    pp.ylabel('Predicted Label',fontsize=8)
    pp.title(freq_band+', '+measure_label)
    #pp.savefig(res_dir+'conf_mat_MLR_'+freq_band+'_'+measure_label, format=fmt_grph)
    pp.tight_layout()
    pp.savefig(res_dir+f'conf_mat_MLR_{classification_feature+1}{greek_band+1}_'+freq_band+'_'+measure_label + '.' + fmt_grph)
    pp.close()


    # plot RFE support network
    if 1:
        pp.figure(figsize=[17,5])
        pp.axes([0.05,0.05,0.95,0.95])
        pp.axis('off')
        if i_measure == 0: # power
            list_best_feat = np.argsort(rk_pow[i_band,:,:].mean(0))[:10] # select 10 best features
            node_color_aff = []
            g = nx.Graph()
            for i in range(N):
                g.add_node(i)
                if i in list_best_feat:
                    node_color_aff += ['red']
                else:
                    node_color_aff += ['orange']
            nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
            nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
        else: # covariance or correlation
            list_best_feat = np.argsort(rk_FC[i_band,i_measure-1,:,:].mean(0))[:20] # select 20 best features
            g = nx.Graph()
            for i in range(N):
                g.add_node(i)
            node_color_aff = ['orange']*N
            list_ROI_from_to = [] # list of input/output ROIs involved in connections of support network
            for ij in list_best_feat:
                g.add_edge(col_ind[ij],row_ind[ij])
            nx.draw_networkx_nodes(g,pos=pos_circ,node_color=node_color_aff)
            nx.draw_networkx_labels(g,pos=pos_circ,labels=ch_labels)
            nx.draw_networkx_edges(g,pos=pos_circ,edges=g.edges(),edge_color=listcolours[i_band])
        pp.title(freq_band, fontsize = 30)
        pp.savefig(res_dir+'support_net_RFE_'+freq_band+'_'+measure_label + '.' + fmt_grph,bbox_inches='tight')
        pp.close()

    # plot stability of RFE rankings
    if 1:
        pp.figure(figsize=[4,3])
        pp.axes([0.2,0.2,0.7,0.7])
        pp.violinplot(pearson_corr_rk[i_band,:,:].T,positions=range(n_measures),widths=[0.4]*3)
        pp.axis(ymin=0,ymax=1)
        pp.xticks(range(n_measures),measure_labels,fontsize=8)
        pp.ylabel('Pearson between rankings',fontsize=8)
        pp.title(freq_band)
        pp.savefig(res_dir+'stab_RFE_rankings_'+freq_band + '.' + fmt_grph,bbox_inches='tight')
        pp.close()
                
    # save performance params
    cnf_matrix = conf_matrix[greek_band, classification_feature, :, 0, :,:].mean(0) #mean across n_reps, 0 so only logistic reg
    def performance_params(confusion_matrix):
        FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)  
        FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
        TP = np.diag(cnf_matrix)
        TN = cnf_matrix.sum() - (FP + FN + TP)
        
        FP = FP.astype(float)
        FN = FN.astype(float)
        TP = TP.astype(float)
        TN = TN.astype(float)
        
        # Sensitivity, hit rate, recall, or true positive rate
        TPR = TP/(TP+FN)
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        # Precision or positive predictive value
        PPV = TP/(TP+FP)
        # Negative predictive value
        NPV = TN/(TN+FN)
        # Fall out or false positive rate
        FPR = FP/(FP+TN)
        # False negative rate
        FNR = FN/(TP+FN)
        # False discovery rate
        FDR = FP/(TP+FP)
        # Overall accuracy
        ACC = (TP+TN)/(TP+FP+FN+TN)
        
        perfs = np.concatenate((TPR[np.newaxis,:],TNR[np.newaxis,:],PPV[np.newaxis,:],NPV[np.newaxis,:],FPR[np.newaxis,:],FNR[np.newaxis,:],FDR[np.newaxis,:],ACC[np.newaxis,:]), axis=1)
        np.save(res_dir+'perfs_'+str(classification_feature+1)+str(greek_band+1)+'.npy',perfs)
        
    performance_params(cnf_matrix) 
        
        # pre vs post condition confusion
    if 0:
        overall_cond_cnf = (sum(sum(cnf_matrix[5:9, 0:4])) + sum(sum(cnf_matrix[0:4, 5:9]))) / sum(sum(cnf_matrix))
        np.save(res_dir+'cond_cnf_'+str(classification_feature+1)+str(greek_band+1)+'.npy',overall_cond_cnf)       
        
        
    
    ####################

    # plot roc
    if 1:
        def plot_multiclass_roc(clf, X_test, y_test, n_classes, figsize=(17, 6), save_text = True, legend = True):
            y_score = clf.decision_function(X_test)
        
            # structures
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
        
            # calculate dummies once
            y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
        
            # roc for each class
            fig, ax = pp.subplots(figsize=figsize)
            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('false positive rate',fontsize=8)
            ax.set_ylabel('true positive rate',fontsize=8)
            ax.set_title(freq_band+', '+measure_label)
            #ax.set_title('Receiver operating characteristic',fontsize=8)
            for i in range(n_classes):
                ax.plot(fpr[i], tpr[i], label='%i (auc = %0.2f)' % (i, roc_auc[i]))
            if legend: ax.legend(loc="lower right",frameon=False,handlelength=1)
            pp.legend(frameon=False)
            ax.grid(alpha=.4)
            sns.despine()
            pp.tight_layout()
            pp.savefig(res_dir+f'roc_{classification_feature+1}{greek_band+1}_'+freq_band+'.png')
            if save_text: pd.DataFrame(roc_auc, index=[0]).to_excel(f'auc_{classification_feature+1}{greek_band+1}.xlsx', engine='xlsxwriter')  
            pp.close()
        plot_multiclass_roc(c_MLR_original, vect_features[ind_test_original,:], labels[ind_test_original], n_classes=n_motiv, figsize=(4, 3), legend = False, save_text = False)

