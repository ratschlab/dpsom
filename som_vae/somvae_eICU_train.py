"""
Script to train the SOM-VAE model / evaluate the time series unrolling
"""

import os
import uuid
import shutil
import random
from glob import glob
from datetime import date
import ipdb
import csv
import timeit
import sys

import numpy as np
import numpy.random as nprand

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import pandas as pd
from sklearn.metrics import mean_squared_error
from tqdm import tqdm, trange
import sacred
from sacred.stflow import LogFileWriter
from sklearn.model_selection import train_test_split
from sklearn import metrics
import h5py

from somvae_model import SOMVAE

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

@ex.config
def ex_config():
    """Sacred configuration for the experiment.
    
    Params:
        num_epochs (int): Number of training epochs.
        patience (int): Patience for the early stopping.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        decay_factor (float): Factor for the learning rate decay.
        name (string): Name of the experiment.
        ex_name (string): Unique name of this particular run.
        logdir (path): Directory for the experiment logs.
        modelpath (path): Path for the model checkpoints.
        interactive (bool): Indicator if there should be an interactive progress bar for the training.
        data_set (string): Data set for the training.
        save_model (bool): Indicator if the model checkpoints should be kept after training and evaluation.
        time_series (bool): Indicator if the model should be trained on linearly interpolated
            MNIST time series.
        mnist (bool): Indicator if the model is trained on MNIST-like data.
        validation (bool): If "True" validation set is used for evaluation, otherwise test set is used.
        only_evaluate (bool): If "True" do not train a new model, just evaluate an existing one.
    """
    num_epochs = 100
    patience = 100
    batch_size = 32
    latent_dim = 10
    som_dim = [16,16]
    learning_rate = 0.0005
    alpha = 1.0
    beta = 0.9
    gamma = 1.8
    tau = 1.4
    decay_factor = 0.9
    name = ex.get_experiment_info()["name"]
    only_evaluate=False
    random_seed=0
    
    if only_evaluate:
        ex_name="SOMVAE_hyperopt_256_16-16_2020-02-05_813ed-rseed0"
    else:
        ex_name = "SOMVAE_{}_{}_{}-{}_{}_{}-rseed{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()), uuid.uuid4().hex[:5],random_seed)

    modelpath = "../models/{}/{}.ckpt".format(ex_name, ex_name)
    logdir = "../logs/{}".format(ex_name)
    interactive = True
    save_model = True
    time_series = True
    mnist = False
    validation = False
    more_runs = False
    hours_to_predict=24

    train_ratio=1.00  # Reduce training set size by a factor
    benchmark=False # If true, get training time statistics per epoch!

@ex.capture
def get_data(validation):
    """Load the saved data and split into training, validation and test set.
        Args:
            validation (bool): If "True" validation set is used for evaluation, otherwise test set is used.
        Yields:
            np.array: Training data.
            np.array: Val/test data depending on validation value.
            np.array: Training labels.
            np.array: Val/test data depending on validation value.
            np.array: Val/test labels."""

    hf = h5py.File('../data/eICU_data.csv','r')
    data_total = np.array(hf.get('x'))
    endpoints_total = np.array(hf.get('y'))
    hf.close()
    data_train, data_val, y_train, endpoints_total_val = train_test_split(data_total[:int(len(data_total) * 0.85)],
                                                                          endpoints_total[:int(len(data_total) * 0.85)],
                                                                          test_size=0.20,
                                                                          random_state=42)
    if not validation:
        data_val = data_total[int(len(data_total) * 0.85):]
        endpoints_total_val = endpoints_total[int(len(data_total) * 0.85):]
    return data_train, data_val, y_train, endpoints_total_val


def get_normalized_data(data, patientid, mins, scales):
    return ((data[data['patientunitstayid'] == patientid] - mins) /
            scales).drop(["patientunitstayid", "ts"], axis=1).fillna(0).values


@ex.capture
def batch_generator(data_train, data_val, endpoints_total_val, batch_size, mode="train"):
    """Generator for the data batches.
        Args:
            data_train: training set.
            data_val: validation/test set.
            labels_val: labels of the validation set.
            batch_size (int): Batch size for the training.
            mode (str): Mode in ['train', 'val', 'test'] that decides which data set the generator
                samples from (default: 'train').
        Yields:
            np.array: Data batch.
            np.array: Labels batch.
            int: Offset of the batch in dataset.
    """
    while True:
        if mode == "train":
            for i in range(len(data_train) // batch_size):
                time_series = data_train[i * batch_size: (i + 1) * batch_size]
                yield time_series
        elif mode == "val":
            for i in range(len(data_val) // batch_size):
                time_series = data_val[i * batch_size: (i + 1) * batch_size]
                time_series_endpoint = endpoints_total_val[i * batch_size: (i + 1) * batch_size]
                yield time_series, time_series_endpoint
        else:
            raise ValueError("The mode has to be in {train, val}")


@ex.capture
def train_model(model, data_train, data_val, endpoints_total_val, lr_val, num_epochs, patience, batch_size, logdir,
                modelpath, learning_rate, interactive, only_evaluate, benchmark, train_ratio):
    """Trains the SOM-VAE model.
    
    Args:
        model (SOM-VAE): SOM-VAE model to train.
        x (tf.Tensor): Input tensor or placeholder.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of epochs to train.
        patience (int): Patience parameter for the early stopping.
        batch_size (int): Batch size for the training generator.
        logdir (path): Directory for saving the logs.
        modelpath (path): Path for saving the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        interactive (bool): Indicator if we want to have an interactive
            progress bar for training.
        generator (generator): Generator for the data batches.
        only_evaluate (bool): Do not actually perform train but just load the model
    """

    len_data_val = len(data_val)
    val_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="val")
    x = model.inputs
    
    if benchmark:
        times_per_epoch=[]
    
    # Train the model
    if not only_evaluate:
        len_data_train = len(data_train)
        num_batches = len_data_train // batch_size
        train_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="train")
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
        summaries = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            patience_count = 0
            test_losses = []
            with LogFileWriter(ex):
                train_writer = tf.summary.FileWriter(logdir+"/train", sess.graph)
                test_writer = tf.summary.FileWriter(logdir+"/test", sess.graph)
            print("Training...")
            train_step_SOMVAE, train_step_prob = model.optimize
            if interactive:
                pbar = tqdm(total=num_epochs * (num_batches))
            if benchmark:
                t_begin_all=timeit.default_timer()

            for epoch in range(num_epochs):

                if benchmark:
                    t_begin=timeit.default_timer()

                batch_val, _ = next(val_gen)
                test_loss, summary = sess.run([model.loss, summaries], feed_dict={x: batch_val, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
                test_losses.append(test_loss)
                test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                if test_losses[-1] == min(test_losses):
                    saver.save(sess, modelpath, global_step=epoch)
                    patience_count = 0
                else:
                    patience_count += 1
                if patience_count >= patience:
                    break
                for i in range(num_batches):
                    batch_data = next(train_gen)
                    if i % 100 == 0:
                        train_loss, summary = sess.run([model.loss, summaries], 
                                                       feed_dict={x: batch_data, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    train_step_SOMVAE.run(feed_dict={x: batch_data, lr_val: learning_rate, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
                    train_step_prob.run(feed_dict={x: batch_data, lr_val: learning_rate * 100, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
                    if interactive:
                        pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                        pbar.update(1)

                if benchmark:
                    t_end=timeit.default_timer()
                    times_per_epoch.append(t_end-t_begin)

            if benchmark:
                t_end_all=timeit.default_timer()
                ttime_all=t_end_all-t_begin_all

            saver.save(sess, modelpath)
            pbar.close()

    if benchmark:
        print("Total time series: {}/{}".format(train_ratio, len(data_train)))
        print("Fitting time per epoch: {:.3f}".format(np.mean(times_per_epoch)))
        print("Total fitting time: {:.3f}".format(ttime_all))
        sys.exit(0)
            
    # Evaluate the model in any case
    with tf.Session() as sess:
        results = evaluate_model(model, x, val_gen, len_data_val, modelpath)
    return results


@ex.capture
def evaluate_model(model, x, val_gen, len_data_val, modelpath, batch_size, som_dim, hours_to_predict):
    """Evaluates the performance of the trained model in terms of normalized
    mutual information, purity and mean squared error.
    
    Args:
        model (SOM-VAE): Trained SOM-VAE model to evaluate.
        x (tf.Tensor): Input tensor or placeholder.
        modelpath (path): Path from which to restore the model.
        batch_size (int): Batch size for the evaluation.
        
    Returns:
        dict: Dictionary of evaluation results (NMI, Purity, MSE).
    """
    saver = tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
    num_batches = len_data_val // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)
        test_k_all = []
        test_rec_all = []
        test_mse_all = []
        labels_val_all = []
        target_idx=[(i,j) for i in range(som_dim[0]) for j in range(som_dim[1])]
        assert(len(target_idx)==256)

        print("Evaluation...")

        all_gt_postfix=[]
        all_postfix=[]

        for i in range(num_batches):
            print("Evaluating batch: {}/{}".format(i,num_batches))
            batch_data, batch_labels = next(val_gen)
            tprob=sess.run(model.transition_probabilities) # Extract the markov transition matrix

            batch_data_blanked=batch_data.copy()
            batch_data_blanked[:,72-hours_to_predict:,:]=np.zeros((hours_to_predict,98))

            cluster_ass=sess.run(model.k, feed_dict={x: batch_data_blanked, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
            cass=np.reshape(cluster_ass,(batch_size,-1))
            last_state=cass[:,72-hours_to_predict-1]
            last_state_k1=last_state // som_dim[0]
            last_state_k2=last_state % som_dim[1]
            postfix=np.zeros((batch_size,hours_to_predict,98))
            gt_postfix=batch_data[:,72-hours_to_predict:,:]

            for ridx in range(batch_size):
                cur_state=(last_state_k1[ridx],last_state_k2[ridx])
                for cidx in range(hours_to_predict):
                    next_state=tprob[cur_state[0],cur_state[1],:,:]
                    target_probs=[next_state[i,j] for i in range(som_dim[0]) for j in range(som_dim[1])]
                    cur_state=random.choices(target_idx,weights=target_probs,k=1)[0]
                    k_state=cur_state[0]*som_dim[0]+cur_state[1]
                    dec_state=sess.run(model.reconstruction_q, feed_dict={x: np.zeros_like(batch_data), model.is_training: False, model.prediction_input: np.array([k_state])})
                    postfix[ridx,cidx,:]=dec_state.flatten()

            all_gt_postfix.append(gt_postfix)
            all_postfix.append(postfix)
            test_k_all.extend(cluster_ass)
            test_rec = sess.run(model.reconstruction_q, feed_dict={x: batch_data, model.is_training: True, model.prediction_input: np.zeros(batch_size)})
            test_rec_all.extend(test_rec)
            test_mse_all.append(mean_squared_error(test_rec.flatten(), batch_data.flatten()))
            labels_val_all.extend(batch_labels)

        gt_last=np.concatenate(all_gt_postfix,axis=0)
        pred_last=np.concatenate(all_postfix,axis=0)
        gt_last=np.reshape(gt_last,(gt_last.shape[0]*gt_last.shape[1],-1))[:9000] # To make sure same data as T-DPSOM is used
        pred_last=np.reshape(pred_last,(pred_last.shape[0]*pred_last.shape[1],-1))[:9000] # To make sure same data as T-DPSOM is used
        print("Shape of ground truth and predictions of post-fix: {}".format(gt_last.shape))
        somvae_pred_mse=metrics.mean_squared_error(pred_last,gt_last)
        labels_val_all = np.array(labels_val_all)
        test_k_all = np.array(test_k_all)
        labels_val_all = np.reshape(labels_val_all, (-1, labels_val_all.shape[-1]))
        NMI_24 = metrics.normalized_mutual_info_score(labels_val_all[:, 3], test_k_all)
        NMI_12 = metrics.normalized_mutual_info_score(labels_val_all[:, 2], test_k_all)
        NMI_6 = metrics.normalized_mutual_info_score(labels_val_all[:, 1], test_k_all)
        NMI_1 = metrics.normalized_mutual_info_score(labels_val_all[:, 0], test_k_all)
        AMI_1 = metrics.adjusted_mutual_info_score(test_k_all, labels_val_all[:, 0])

    results = {}
    results["NMI_24"] = NMI_24
    results["NMI_12"] = NMI_12
    results["NMI_6"] = NMI_6
    results["NMI_1"] = NMI_1
    results["PRED_MSE"]=somvae_pred_mse

    return results
 

@ex.automain
def main(latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, tau, modelpath, save_model, 
         mnist,ex_name, more_runs, random_seed, num_epochs, train_ratio):
    """Main method to build a model, train it and evaluate it.
    
    Args:
        latent_dim (int): Dimensionality of the SOM-VAE's latent space.
        som_dim (list): Dimensionality of the SOM.
        learning_rate (float): Learning rate for the training.
        decay_factor (float): Factor for the learning rate decay.
        alpha (float): Weight for the commitment loss.
        beta (float): Weight for the SOM loss.
        gamma (float): Weight for the transition probability loss.
        tau (float): Weight for the smoothness loss.
        modelpath (path): Path for the model checkpoints.
        save_model (bool): Indicates if the model should be saved after training and evaluation.
        train_ratio (float): Ratio of training data that should be used, to test performance vs. train data size
        
    Returns:
        dict: Results of the evaluation (NMI for the different APACHE labels, as well as prediction MSE of the last 6 steps).
    """

    random.seed(random_seed)
    nprand.seed(random_seed)

    # Dimensions for MNIST-like data
    input_length = 28
    input_channels = 98
    lr_val = tf.placeholder_with_default(learning_rate, [])

    model = SOMVAE(latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, decay_factor=decay_factor,
            input_length=input_length, input_channels=input_channels, alpha=alpha, beta=beta, gamma=gamma,
            tau=tau, mnist=mnist)

    data_train, data_val, _, endpoints_total_val = get_data()

    if train_ratio<1.0:
        data_train=data_train[:int(len(data_train)*train_ratio)]

    if not more_runs:
        results = train_model(model, data_train, data_val, endpoints_total_val, lr_val)

        with open("../results/SOMVAE_evaluation_eICU_{}.tsv".format(random_seed), "w") as fp:
            csv_fp=csv.writer(fp,delimiter='\t')
            csv_fp.writerow(["seed", "n_epochs", "NMI24", "NMI12", "NMI6", "NMI1", "PRED_MSE"])
            csv_fp.writerow([str(random_seed), str(num_epochs), str(results["NMI_24"]),
                             str(results["NMI_12"]), str(results["NMI_6"]),
                             str(results["NMI_1"]), str(results["PRED_MSE"])])

    else:
        NMI_24_all = []
        NMI_12_all = []
        NMI_6_all = []
        NMI_1_all = []
        PRED_MSE_all=[]
        T = 1             # Was 10, changed to allow parallel execution
        for i in range(T):
            results = train_model(model, data_train, data_val, endpoints_total_val, lr_val)
            NMI_24_all.append(results["NMI_24"])
            NMI_12_all.append(results["NMI_12"])
            NMI_6_all.append(results["NMI_6"])
            NMI_1_all.append(results["NMI_1"])
            PRED_MSE_all.append(results["PRED_MSE"])

        NMI_24_mean = np.mean(NMI_24_all)
        NMI_24_sd = np.std(NMI_24_all)
        NMI_12_mean = np.mean(NMI_12_all)
        NMI_12_sd = np.std(NMI_12_all)
        NMI_6_mean = np.mean(NMI_6_all)
        NMI_6_sd = np.std(NMI_6_all)
        NMI_1_mean = np.mean(NMI_1_all)
        NMI_1_sd = np.std(NMI_1_all)
        PRED_MSE_mean=np.mean(PRED_MSE_all)
        PRED_MSE_sd=np.std(PRED_MSE_ALL)



    if not save_model:
        shutil.rmtree(os.path.dirname(modelpath))

    return results


