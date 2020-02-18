''' Further experiments for comparing T-DPSOM against the baselines
    on the eICU data-set'''

import random
import argparse
import ipdb
import os
import os.path
import sys
import timeit
import csv

import warnings
warnings.filterwarnings('ignore')

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import numpy as np
import numpy.random as nprand
import h5py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import sklearn
from sklearn import metrics
import seaborn as sns

import sklearn.cluster as skcluster
import sklearn.decomposition as skdecomp
import sklearn.linear_model as sklm
import sklearn.preprocessing as skpp
import sklearn.neural_network as sknn

import hmmlearn.hmm as hmm
import lightgbm as lgbm

from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout

def get_data(test=True):
    ''' Get the precomputed data from the file-system'''
    hf = h5py.File(configs["eicu_data"], 'r')
    data_total = np.array(hf.get('x'))
    endpoints_total = np.array(hf.get('y'))
    hf.close()
    data_train, data_val, y_train, endpoints_total_val = train_test_split(data_total[:int(len(data_total) * 0.85)],
                                                                          endpoints_total[:int(len(data_total) * 0.85)],
                                                                          test_size=0.20,
                                                                          random_state=42)
    if test:
        data_val = data_total[int(len(data_total) * 0.85):]
        endpoints_total_val = endpoints_total[int(len(data_total) * 0.85):]
    return data_train, data_val, y_train, endpoints_total_val


def batch_generator(data_train, data_val, endpoints_total_val, batch_size, mode="train"):
    while True:
        if mode == "train":
            for i in range(len(data_train) // batch_size):
                time_series = data_train[i * batch_size: (i + 1) * batch_size]
                yield time_series,i
        elif mode == "val":
            for i in range(len(data_val) // batch_size):
                time_series = data_val[i * batch_size: (i + 1) * batch_size]
                time_series_endpoint = endpoints_total_val[i * batch_size: (i + 1) * batch_size]
                yield time_series, time_series_endpoint,i
        else:
            raise ValueError("The mode has to be in {train, val}")

def z_dist_flat(z_e, embeddings, som_dim, latent_dim):
    """Computes the distances between the encodings and the embeddings."""
    emb = np.reshape(embeddings, (som_dim[0]*som_dim[1], -1))
    z = np.reshape(z_e, (z_e.shape[0], 1, latent_dim))
    z = np.tile(z, [1,som_dim[0]*som_dim[1], 1])
    z_dist = np.square(z-emb)
    z_dist_red = np.sum(z_dist, axis=-1)
    return z_dist_red

        
def execute(configs):
    ''' Main script'''
    ex_name=configs["job_id"]
    batch_size=configs["batch_size"]
    random.seed(configs["random_state"])
    max_n_step=72

    # Get the data, if test=False the data_val is the validation data, if test=True the data_val is the test data:
    modelpath = "../models/{}/{}".format(ex_name, ex_name)
    data_train, data_val, endpoints_total_train, endpoints_total_val = get_data(test=True)

    # Fit a HMM model

    if configs["train_hmm"]:
        tss=[data_train[i,:,:] for i in range(data_train.shape[0])]
        random.shuffle(tss)
        tss=tss[:configs["hmm_train_subset"]]
        X_flat=np.concatenate(tss,axis=0)
        hmm_model=hmm.GaussianHMM(configs["hmm_n_states"],covariance_type="diag", 
                              random_state=configs["random_state"],verbose=True)
        t_begin=timeit.default_timer()
        hmm_model=hmm_model.fit(X_flat,lengths=[72]*len(tss))
        t_end=timeit.default_timer()
        print("Seconds to fit HMM: {:.3f}".format(t_end-t_begin))

    len_data_val = len(data_val)
    data_train_flat=np.reshape(data_train, (data_train.shape[0]*data_train.shape[1],data_train.shape[2]))
    data_val_flat=np.reshape(data_val, (data_val.shape[0]*data_val.shape[1],data_val.shape[2]))

    print("Fitting K-means...")

    if configs["train_kmeans"]:
        kmeans_model=skcluster.MiniBatchKMeans(n_clusters=configs["km_nclusters"],random_state=configs["random_state"])
        kmeans_model.fit(data_train_flat)
        cval_km=kmeans_model.predict(data_val_flat)
        dist_km=kmeans_model.transform(data_val_flat)
        cval_km=np.expand_dims(cval_km,axis=1)
        ohenc=skpp.OneHotEncoder(categories=[list(range(configs["km_nclusters"]))],sparse=False)
        cval_oh=ohenc.fit_transform(cval_km)

    # Save the labels:
    labels_val_all = endpoints_total_val
    labels_val_all = np.array(labels_val_all)

    # Worst APACHE score right now
    labels_1 = labels_val_all[:,:,0]
    labels_6 = labels_val_all[:,:,1]
    labels_12 = labels_val_all[:,:,2]    
    labels_24 = labels_val_all[:,:,3]

    # Dynamic mortality in next hours hospital
    hosp_disc_1 = labels_val_all[:,:,4]
    hosp_disc_6 = labels_val_all[:,:,5]
    hosp_disc_12 = labels_val_all[:,:,6]
    hosp_disc_24 = labels_val_all[:,:,7]

    # Dynamic mortality in next hours ICU
    u_disc_1 = labels_val_all[:,:,8]
    u_disc_6 = labels_val_all[:,:,9]
    u_disc_12 = labels_val_all[:,:,10]
    u_disc_24 = labels_val_all[:,:, 11]
    
    patient_id = labels_val_all[:,:,-1]
    
    labels_1 = labels_1.astype(int)
    labels_6 = labels_6.astype(int)
    labels_12 = labels_12.astype(int)
    labels_24 = labels_24.astype(int)
    hosp_disc_12 = hosp_disc_12.astype(int)
    hosp_disc_24 = hosp_disc_24.astype(int)
    hosp_disc_1 = hosp_disc_1.astype(int)
    hosp_disc_6 = hosp_disc_6.astype(int)
    u_disc_1 = u_disc_1.astype(int)
    u_disc_6 = u_disc_6.astype(int)
    u_disc_12 = u_disc_12.astype(int)
    u_disc_24 = u_disc_24.astype(int)

    if configs["tasks"]=="bclf":
        labels_cat = {"h1": hosp_disc_1.flatten(),
                      "h6": hosp_disc_6.flatten(),
                      "h12": hosp_disc_12.flatten(),
                      "h24": hosp_disc_24.flatten(),
                      "u1": u_disc_1.flatten(),
                      "u6": u_disc_6.flatten(),
                      "u12": u_disc_12.flatten(),
                      "u24": u_disc_24.flatten()}
    elif configs["tasks"]=="reg":
        labels_cat = {"l1": labels_1.flatten(),
                      "l6": labels_6.flatten(),
                      "l12": labels_12.flatten(),
                      "l24": labels_24.flatten()}
    else:
        assert(False)

    # Shape of the labels (num_patients, num_timestamps):
    print("Shape of labels: {}".format(patient_id.shape))
    
    # ## Create heat-maps, trajectories and probability distributions
    som_dim = [configs["som_dim"],configs["som_dim"]]
    latent_dim=10

    # Test the model
    tf.reset_default_graph()
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.import_meta_graph(modelpath+".meta")
        saver.restore(sess, modelpath)
        graph = tf.get_default_graph()
        k = graph.get_tensor_by_name("k/k:0")
        z_e = graph.get_tensor_by_name("z_e_sample/z_e:0")
        #next_z_e = graph.get_tensor_by_name("prediction/next_z_e:0")
        x = graph.get_tensor_by_name("inputs/x:0")
        is_training = graph.get_tensor_by_name("is_training/is_training:0")
        graph = tf.get_default_graph()
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        state1 = graph.get_tensor_by_name("prediction/next_state/next_state:0")
        q = graph.get_tensor_by_name("q/distribution/q:0")
        embeddings = graph.get_tensor_by_name("embeddings/embeddings:0")
        reconstruction = graph.get_tensor_by_name("reconstruction_e/x_hat:0")
        z_q = graph.get_tensor_by_name("z_q/z_q:0")

        print("Evaluation...")
        training_dic = {is_training: True, z_e_p: np.zeros((72 * len(data_val), latent_dim)), 
                        init_1: np.zeros((2, len(data_val), 10))}

        val_gen = batch_generator(data_train, data_val, endpoints_total_val, batch_size, mode="val")
        num_batches=len_data_val//batch_size
        test_k_all=[]
        test_qq_all=[]
        test_z_e_all=[]
        test_z_q_all=[]
        labels_sanity=[]

        for i in range(num_batches):
            batch_data,batch_labels,ii=next(val_gen)
            batch_labels_san=np.reshape(batch_labels, (-1, batch_labels.shape[-1]))[:,3]
            batch_k_all=sess.run(k,feed_dict={x: batch_data})
            batch_k_all=batch_k_all.reshape((-1,72))
            batch_qq_all=sess.run(q, feed_dict={x:batch_data})
            batch_qq_all=batch_qq_all.reshape((-1,72,configs["som_dim"]*configs["som_dim"]))
            batch_z_e_all=sess.run(z_e, feed_dict={x:batch_data})
            batch_z_e_all=batch_z_e_all.reshape((-1,72,latent_dim))
            batch_z_q_all=sess.run(z_q, feed_dict={x:batch_data})
            batch_z_q_all=batch_z_q_all.reshape((-1,72,latent_dim))
            test_k_all.extend(batch_k_all)
            test_qq_all.extend(batch_qq_all)
            test_z_e_all.extend(batch_z_e_all)
            test_z_q_all.extend(batch_z_q_all)
            labels_sanity.extend(batch_labels_san)
            
        k_all=np.array(test_k_all)
        qq_all=np.array(test_qq_all)
        z_e_all=np.array(test_z_e_all)
        z_q_all=np.array(test_z_q_all)
        labels_sanity=np.array(labels_sanity).astype(np.int)
        prefix=labels_24.flatten()[:108000]
        assert((labels_sanity==prefix).all()) 

    # z_q= closest centroids in the hidden space for every patient and every time-step
    print("z_q_all shape: {}".format(z_q_all.shape))

    # z_e= hidden state of every patient and e
    print("z_e_all shape: {}".format(z_e_all.shape))

    # k_all= closest centroids index for every patient and every time-step
    print("k_all shape: {}".format(k_all.shape))

    fprob_int=np.expand_dims(k_all.flatten(),axis=1)
    ohenc=skpp.OneHotEncoder(categories=[list(range(configs["som_dim"]*configs["som_dim"]))],sparse=False)
    fprob=ohenc.fit_transform(fprob_int)

    # qq_all= centroid probabilities for every patient and every time-step
    print("qq_all_shape: {}".format(qq_all.shape))

    cprob=np.reshape(qq_all,(qq_all.shape[0]*qq_all.shape[1], qq_all.shape[2]))
    hprob=np.reshape(z_e_all,(z_e_all.shape[0]*z_e_all.shape[1], z_e_all.shape[2]))

    if configs["eval_tasks"]:

        print("Evaluating downstream tasks...")

        all_reprs={"dpsom_oh": fprob, "dpsom_prob": cprob, "dprob_cont": hprob, 
                   "kmeans_dist": dist_km, "kmeans_oh": cval_oh}

        if configs["only_nmi"]:
            all_reprs={"dpsom_int": fprob_int, "kmeans_int": cval_km}

        scale_model=skpp.StandardScaler()

        if configs["ml_model"]=="gbm":
            hp_grid=[8,16,32,64,128]
        else:
            hp_grid=[1.0,0.1,0.01,0.001,0.0001,0.00001]

        with open("../results/mh_results_rseed_{}.tsv".format(configs["random_state"]),'w') as fp:
            csv_fp=csv.writer(fp,delimiter='\t')
            csv_fp.writerow(["repr","task","nmi"])

        for rep in all_reprs.keys():
            repnum=all_reprs[rep][:108000]
            for task in labels_cat.keys():
                tasknum=labels_cat[task][:108000]

                if configs["only_nmi"]:
                    with open("../results/mh_results_rseed_{}.tsv".format(configs["random_state"]),'a') as fp:
                        csv_fp=csv.writer(fp,delimiter='\t')
                        csv_fp.writerow([rep,task,str(metrics.normalized_mutual_info_score(tasknum.flatten(), repnum.flatten()))])
                        print("Repr: {}, Task: {}, NMI: {:.3f}".format(rep,task,metrics.normalized_mutual_info_score(tasknum.flatten(), repnum.flatten())))
                    continue

                trainX=repnum[:int(repnum.shape[0]*0.6),:]

                if not configs["ml_model"]=="gbm":
                    trainX=scale_model.fit_transform(trainX)

                trainy=tasknum[:int(repnum.shape[0]*0.6)]
                valX=repnum[int(repnum.shape[0]*0.6):int(repnum.shape[0]*0.8),:]

                if not configs["ml_model"]=="gbm":
                    valX=scale_model.transform(valX)

                valy=tasknum[int(repnum.shape[0]*0.6):int(repnum.shape[0]*0.8)]
                testX=repnum[int(repnum.shape[0]*0.8):,:]

                if not configs["ml_model"]=="gbm":
                    testX=scale_model.transform(testX)

                testy=tasknum[int(repnum.shape[0]*0.8):]
                best_score=-np.inf

                for alpha_cand in hp_grid:

                    if "l" in task:

                        if configs["ml_model"]=="lr":
                            ds_model=sklm.SGDRegressor(random_state=configs["random_state"], alpha=alpha_cand)
                        elif configs["ml_model"]=="mlp":
                            ds_model=sknn.MLPRegressor(hidden_layer_sizes=(20),alpha=alpha_cand,random_state=configs["random_state"])
                        elif configs["ml_model"]=="gbm":
                            ds_model=lgbm.LGBMRegressor(n_estimators=5000, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, random_state=configs["random_state"],
                                                        num_leaves=alpha_cand,verbose=-1,silent=True)

                    else:

                         if configs["ml_model"]=="lr":
                             ds_model=sklm.SGDClassifier(loss="log", random_state=configs["random_state"],class_weight="balanced",alpha=alpha_cand)
                         elif configs["ml_model"]=="mlp":
                             ds_model=sknn.MLPClassifier(hidden_layer_sizes=(20), alpha=alpha_cand, random_state=configs["random_state"])
                         elif configs["ml_model"]=="gbm":
                             ds_model=lgbm.LGBMClassifier(n_estimators=5000, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, random_state=configs["random_state"],
                                                          verbose=-1,silent=True,num_leaves=alpha_cand)

                    if configs["ml_model"]=="gbm":

                        with suppress_stdout():
                            ds_model.fit(trainX,trainy, eval_set=[(valX,valy)], eval_metric="auc", early_stopping_rounds=20)

                    else:
                        ds_model.fit(trainX,trainy)

                    if "l" in task:
                        pred_scores=ds_model.predict(valX)
                        val_metric=-metrics.mean_absolute_error(valy,pred_scores)
                    else:
                        pred_scores=ds_model.predict_proba(valX)[:,1]
                        val_metric=metrics.roc_auc_score(valy,pred_scores)

                    if val_metric>best_score:
                        best_alpha=alpha_cand
                        best_score=val_metric

                if "l" in task:

                    if configs["ml_model"]=="lr":
                        ds_model=sklm.SGDRegressor(random_state=configs["random_state"], alpha=best_alpha)
                    elif configs["ml_model"]=="mlp":
                        ds_model=sknn.MLPRegressor(hidden_layer_sizes=(20), random_state=configs["random_state"],alpha=best_alpha)
                    elif configs["ml_model"]=="gbm":
                        ds_model=lgbm.LGBMRegressor(n_estimators=5000, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, random_state=configs["random_state"],
                                                    num_leaves=best_alpha,verbose=-1,silent=True)

                else:

                    if configs["ml_model"]=="lr":
                        ds_model=sklm.SGDClassifier(loss="log", random_state=configs["random_state"],class_weight="balanced",alpha=best_alpha)
                    elif configs["ml_model"]=="mlp":
                        ds_model=sknn.MLPClassifier(hidden_layer_sizes=(20), random_state=configs["random_state"],alpha=best_alpha)
                    elif configs["ml_model"]=="gbm":
                        ds_model=lgbm.LGBMClassifier(n_estimators=5000, subsample=0.5, subsample_freq=1, colsample_bytree=0.5, random_state=configs["random_state"],
                                                     num_leaves=best_alpha,verbose=-1,silent=True)

                if configs["ml_model"]=="gbm":

                    with suppress_stdout():
                        ds_model.fit(trainX, trainy, eval_set=[(valX,valy)], eval_metric="auc", early_stopping_rounds=20)

                else:
                    ds_model.fit(trainX,trainy)

                if "l" in task:
                    pred_scores=ds_model.predict(testX)
                    test_metric=metrics.mean_absolute_error(valy, pred_scores)
                else:
                    pred_scores=ds_model.predict_proba(testX)[:,1]
                    test_metric=metrics.roc_auc_score(testy,pred_scores)
                print("Repr {}, Task: {}, AUC/MAE: {:.3f}".format(rep, task, test_metric))

    # Heatmap with respect to current APACHE score
    
    print("Labels shape: {}".format(labels_1.shape))
    labels = np.reshape(labels_1,(-1))
    ones = np.ones((len(np.reshape(k_all, (-1)))))
    clust_matr1 = np.zeros(som_dim[0]*som_dim[1])
    
    for i in range(som_dim[0]*som_dim[1]):
        s1 = np.sum(labels[np.where(np.reshape(k_all, (-1))==i)]) / np.sum(ones[np.where(np.reshape(k_all, (-1))==i)])
        clust_matr1[i] = s1
        
    clust_matr1 = np.reshape(clust_matr1, (som_dim[0],som_dim[1]))
    ax = sns.heatmap(clust_matr1, cmap="YlGnBu", vmax=7)
    plt.savefig(os.path.join(configs["plot_path"], "apache_heatmaps.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(configs["plot_path"], "apache_heatmaps.png"),bbox_inches="tight")
    plt.clf()

    # ICU mortality risk in the next 24 hours:
    labels = np.reshape(u_disc_24,(-1))
    ones = np.ones((len(np.reshape(k_all, (-1)))))
    clust_matr1 = np.zeros(som_dim[0]*som_dim[1])
    
    for i in range(som_dim[0]*som_dim[1]):
        s1 = np.sum(labels[np.where(np.reshape(k_all, (-1))==i)]) / np.sum(ones[np.where(np.reshape(k_all, (-1))==i)])
        clust_matr1[i] = s1
        
    clust_matr1 = np.reshape(clust_matr1, (som_dim[0],som_dim[1]))
    ax = sns.heatmap(clust_matr1, cmap="YlGnBu")
    
    plt.savefig(os.path.join(configs["plot_path"], "mortality_heatmap_24_hours.pdf"),bbox_inches="tight")
    plt.savefig(os.path.join(configs["plot_path"], "mortality_heatmap_24_hours.png"),bbox_inches="tight")
    plt.clf()

    # Trajectories:

    if configs["plot_trajectories"]:

        T = []

        for i in range(1000):
            h = np.reshape(u_disc_1, (-1,72))
            if np.max(h[i]) == 1:
                T.append(i)

        ind_r = np.random.random_integers(0, 50, 10)
        ind_s = np.random.random_integers(0, 500, 10)
        T = np.array(T)
        a = np.concatenate([ind_s, T[ind_r]])

        labels = u_disc_24
        it = 0
        fig, ax = plt.subplots(5, 4, figsize=(50,43)) 
        ones = np.ones((len(np.reshape(k_all, (-1)))))
        clust_matr1 = np.zeros(64)
        clust_matr2 = np.zeros(64)

        for i in range(64):
            s1 = np.sum(labels[np.where(np.reshape(k_all, (-1)) == i)]) / np.sum(ones[np.where(np.reshape(k_all, (-1))==i)])
            clust_matr1[i] = s1

        clust_matr1 = np.reshape(clust_matr1, (8,8))

        for t in a:
            #fig, ax = plt.subplots(figsize=(10,7.5)) 
            if it > 9:
                c = "r"
                #print(t)
            else:
                c = "g"
            cc = it % 4
            rr = it // 4
            g = sns.heatmap(clust_matr1, cmap="YlGnBu",ax=ax[rr][cc])
            som_dim=[8,8]
            k_1 = k_all[t] // som_dim[1]
            k_2 = k_all[t] % som_dim[1]
            ax[rr][cc].plot(k_2[:] + 0.5, k_1[:] + 0.5, color=c, linewidth=4)
            ax[rr][cc].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color=c, s=200, label='Start')
            ax[rr][cc].scatter(k_2[1:-1] + 0.5, k_1[1:-1] + 0.5, color=c, linewidth=5, marker='.')
            ax[rr][cc].scatter(k_2[-1] + 0.5, k_1[-1] + 0.5, color=c, s=500, linewidth=4, marker='x', label='End')
            ax[rr][cc].legend(loc=2, prop={'size': 20})
            it +=1

        plt.savefig(os.path.join(configs["plot_path"], "trajectories.pdf"))
        plt.savefig(os.path.join(configs["plot_path"], "trajectories.png"))
        plt.clf()


    if configs["plot_trajectory_uncertainty"]:

        prob_q = np.reshape(qq, (-1, 72, 64)) 
        i = np.random.randint(0, 50) 
        it = 0
        fig, ax = plt.subplots(2, 3, figsize=(50,25))

        for t in [0, 17, 40, 57, 64, 71]:
            cc = it % 3
            rr = it // 3
            k_1 = k_all[i] // som_dim[1]
            k_2 = k_all[i] % som_dim[1]
            c = "black"
            g1 = sns.heatmap(np.reshape(prob_q[i, t], (8,8)), cmap='Reds', alpha=1,  ax=ax[rr][cc])
            ax[rr][cc].plot(k_2[:] + 0.5, k_1[:] + 0.5, color=c, linewidth=6)
            ax[rr][cc].scatter(k_2[0] + 0.5, k_1[0] + 0.5, color=c, s=800, label='Start')
            ax[rr][cc].scatter(k_2[1:-1] + 0.5, k_1[1:-1] + 0.5, color=c, linewidth=10, marker='.')
            ax[rr][cc].scatter(k_2[-1] + 0.5, k_1[-1] + 0.5, color=c, s=1200, linewidth=10, marker='x', label='End')
            ax[rr][cc].legend(loc=2, prop={'size': 30})  
            ax[rr][cc].set_title("Time-step = {}".format(it*14), fontsize=40)
            it +=1

        plt.savefig(os.path.join(configs["plot_path"], "prob_dist_trajectory.pdf"))
        plt.savefig(os.path.join(configs["plot_path"], "prob_dist_trajectory.png"))
        plt.clf()

    if configs["predict_future_experiment"]:
        seq_prefix=data_val[:,:-6]
        seq_postfix=data_val[:,-6:]
        h_tmat=hmm_model.transmat_
        h_means=hmm_model.means_
        h_cov=hmm_model.covars_
        pred_hat=np.zeros((seq_prefix.shape[0],6,seq_prefix.shape[2]))

        # Predict postfixes
        for pat in range(seq_prefix.shape[0]):
            last_state=hmm_model.decode(seq_prefix[pat,:,:])[1][-1]
            for j in range(6):
                nstate=nprand.choice(range(configs["hmm_n_states"]),p=h_tmat[last_state,:])
                pred_hat[pat,j,:]=nprand.multivariate_normal(h_means[nstate,:], h_cov[nstate,:])

        # Decode entire sequence
        alls=[]
        for pat in range(data_val.shape[0]):
            all_states=hmm_model.decode(data_val[pat,:,:])[1]
            alls.append(all_states)
        pred_k_hmm=np.stack(alls,axis=0).flatten()

        if not configs["debug_mode"]:
            with open("../results/mh_hmm_NMI_results_{}.tsv".format(configs["random_state"]),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["task","NMI"])

        for task in ["l1","l6","l12","l24"]:
            nmi_score=metrics.normalized_mutual_info_score(labels_cat[task].flatten(),pred_k_hmm)
            if not configs["debug_mode"]:
                with open("../results/mh_hmm_NMI_results_{}.tsv".format(configs["random_state"]),'a') as fp:
                    csv_fp=csv.writer(fp,delimiter='\t')
                    csv_fp.writerow([task,str(nmi_score)])
        
        pred_hat=np.reshape(pred_hat,(pred_hat.shape[0]*pred_hat.shape[1],pred_hat.shape[2]))
        seq_postfix=np.reshape(seq_postfix,(seq_postfix.shape[0]*seq_postfix.shape[1],seq_postfix.shape[2]))
        hmm_mse=metrics.mean_squared_error(pred_hat,seq_postfix)

        if not configs["debug_mode"]:
            with open("../results/mh_hmm_results_{}.tsv".format(configs["random_state"]),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["mse"])
                csv_fp.writerow([str(hmm_mse)])

        val_gen=batch_generator(data_train,data_val,endpoints_total_val,300,mode="val")
        tf.reset_default_graph()
        num_pred = 6

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(modelpath+".meta")
            saver.restore(sess, modelpath)
            graph = tf.get_default_graph()
            k = graph.get_tensor_by_name("k/k:0")
            z_e = graph.get_tensor_by_name("z_e_sample/z_e:0")
            next_z_e = graph.get_tensor_by_name("prediction/next_z_e:0")
            x = graph.get_tensor_by_name("inputs/x:0")
            is_training = graph.get_tensor_by_name("is_training/is_training:0")
            graph = tf.get_default_graph()
            init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
            z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
            state1 = graph.get_tensor_by_name("prediction/next_state/next_state:0")
            q = graph.get_tensor_by_name("q/distribution/q:0")
            embeddings = graph.get_tensor_by_name("embeddings/embeddings:0")
            z_p= graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")
            reconstruction = graph.get_tensor_by_name("reconstruction_e/x_hat:0")

            print("Evaluation...")
            training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * len(data_val), latent_dim)),
                            init_1: np.zeros((2, batch_size, 10))}

            z_e_all = sess.run(z_e, feed_dict={x: data_val})
            z_e_all = z_e_all.reshape((-1, 72, latent_dim))
            k_all = sess.run(k, feed_dict={x: data_val})
            k_all = k_all.reshape((-1, 72))
            qq = sess.run(q, feed_dict={x: data_val})
            x_rec = sess.run(reconstruction, feed_dict={x: data_val, is_training: True, z_p: np.zeros((max_n_step * len(data_val), latent_dim))})
            t = 72-num_pred
            embeddings = sess.run(embeddings, feed_dict={x: data_val[:, :t, :]})
            embeddings = np.reshape(embeddings,(-1,latent_dim))
            k_eval=[]
            batch_data= data_val[:, :t, :]
            z_e_o = z_e_all[:, :t, :]
            k_o = k_all[:, :t]
            f_dic = {x: batch_data}
            f_dic.update(training_dic)
            next_z_e_o = sess.run(next_z_e, feed_dict=f_dic)
            next_z_e_o = next_z_e_o[:, -1, :]
            state1_o = sess.run(state1, feed_dict=f_dic)
            k_next = np.argmin(z_dist_flat(next_z_e_o, embeddings, som_dim,latent_dim), axis=-1)
            k_o = np.concatenate([k_o, np.expand_dims(k_next,1)], axis=1)
            z_e_o = np.concatenate([z_e_o, np.expand_dims(next_z_e_o, 1)], axis=1)
            f_dic = {x: np.zeros((len(data_val),1, 98)), is_training: False, z_e_p: np.zeros((1 * len(data_val), latent_dim)),
                     z_p: next_z_e_o, init_1: np.zeros((2, batch_size, 10))}
            x_pred_hat = np.reshape(sess.run(reconstruction, feed_dict=f_dic), (-1, 1, 98))

            for i in range(num_pred-1):
                print(i)
                inp = data_val[:, (t + i), :]
                f_dic = {x: np.reshape(inp, (inp.shape[0],1,inp.shape[1]))}
                val_dic = {is_training: False, z_e_p: next_z_e_o, init_1: state1_o, z_p: np.zeros((max_n_step * len(data_val), latent_dim))}
                f_dic.update(val_dic)
                next_z_e_o = sess.run(next_z_e, feed_dict=f_dic)
                state1_o = sess.run(state1, feed_dict=f_dic)
                k_next = np.argmin(z_dist_flat(next_z_e_o, embeddings, som_dim, latent_dim), axis=-1)
                k_o = np.concatenate([k_o, np.expand_dims(k_next,1)], axis=1)
                z_e_o = np.concatenate([z_e_o, next_z_e_o], axis=1)
                next_z_e_o = np.reshape(next_z_e_o, (-1, latent_dim))
                f_dic = {x: np.zeros((len(data_val),1, 98)), is_training: False, z_e_p: np.zeros((max_n_step * len(data_val), latent_dim)),
                     z_p: next_z_e_o, init_1: np.zeros((2, batch_size, 10))}
                final_x = sess.run(reconstruction, feed_dict=f_dic)
                x_pred_hat = np.concatenate([x_pred_hat, np.reshape(final_x, (-1, 1, 98))], axis = 1)

            f_dic = {x: np.zeros((len(data_val),1, 98)), is_training: False, z_e_p: np.zeros((max_n_step * len(data_val), latent_dim)),
                     z_p: z_e_all[:, t-1, :], init_1: np.zeros((2, batch_size, 10))}
            final_x = sess.run(reconstruction, feed_dict=f_dic)

        print("Proposed model MSE: {:.3f}".format(sklearn.metrics.mean_squared_error(np.reshape(x_pred_hat, (-1, 98)), np.reshape(data_val[:, -6:], (-1, 98)))))

        # Accuracy of unrolled state:

        k_true = np.reshape(k_all[:, -num_pred:], (-1))
        k_pred = np.reshape(k_o[:, -num_pred:], (-1))
        tot = 0
        acc = 0
        for i in range(len(k_true)):
            tot += 1
            if k_true[i] == k_pred[i]:
                acc += 1
        acc = acc / tot
        print("Accuracy: {:.3f}".format(acc))

        # SAME-STATE baseline:

        x_same = np.reshape(final_x, (-1,1, 98))
        for i in range(5):
            x_same = np.concatenate([x_same, np.reshape(final_x, (-1,1, 98))], axis=1)
        x_same=np.array(x_same)

        print("Same state baseline MSE: {:.3f}".format(sklearn.metrics.mean_squared_error(np.reshape(x_same, (-1, 98)), np.reshape(data_val[:, -6:], (-1, 98)))))

def parse_cmd_args():

    parser=argparse.ArgumentParser()

    # Modes (activate which one to use)
    parser.add_argument("--plot_trajectories", default=False, action="store_true", help="Plot trajs?")
    parser.add_argument("--plot_trajectory_uncertainty", default=False, action="store_true", help="Plot traj. uncertainty")
    parser.add_argument("--predict_future_experiment", default=False, action="store_true", help="Predict future experiment")
    parser.add_argument("--eval_tasks", default=False, action="store_true", help="Evaluate downstream tasks?")

    # TDP-SOM
    parser.add_argument("--job_id", default=None, help="Model to evaluate") # Replace with name of model folder to be evaluated
    parser.add_argument("--som_dim", default=16, type=int, help="Dimension of SOM grid")
    parser.add_argument("--batch_size", default=300, type=int, help="Batch size to use")

    # Downstream tasks
    parser.add_argument("--tasks", default="reg", help="Should clf or regression task be analyzed?")
    parser.add_argument("--only_nmi", default=True, action="store_true", help="Only evaluate NMI, no downstream")
    parser.add_argument("--ml_model", default="lr", help="Which DS model to use?")

    # K-means baseline
    parser.add_argument("--train_kmeans", default=False, action="store_true", help="Fit K-means baseline")
    parser.add_argument("--km_nclusters", default=256, type=int, help="Number of K-means clusters")
    
    # Hidden Markov model
    parser.add_argument("--train_hmm", default=True,action="store_true")
    parser.add_argument("--hmm_train_subset", type=int, default=None, help="Number of patients to randomly include in train set")
    parser.add_argument("--hmm_n_states", type=int, default=256, help="Number of states in HMM")

    parser.add_argument("--random_state", type=int, default=2020, help="RSEED")
    parser.add_argument("--debug_mode", default=False, action="store_true", help="No output to FS")
    
    # Input paths
    parser.add_argument("--eicu_data", default="../data/eICU_data.csv")
    
    # Output paths
    parser.add_argument("--plot_path", default="../data/plots", help="Plotting base path")

    configs=vars(parser.parse_args())
    return configs

if __name__=="__main__":
    configs=parse_cmd_args()
    execute(configs)

