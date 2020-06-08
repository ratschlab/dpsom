"""
Script for training the TempDPSOM model
"""

import uuid
import sys
import timeit
from datetime import date
import numpy as np

try:
    import tensorflow.compat.v1 as tf 
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from tqdm import tqdm
import sacred
from sacred.stflow import LogFileWriter
import math
import h5py
from sklearn import metrics
from TempDPSOM_model import TDPSOM
from sklearn.model_selection import train_test_split
import sklearn

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs_eICU"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

@ex.config
def ex_config():
    """Sacred configuration for the experiment.
        Params:
            input_size (int): Length of the input vector.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for the training.
            latent_dim (int): Dimensionality of the T-DPSOM's latent space.
            som_dim (list): Dimensionality of the self-organizing map.
            learning_rate (float): Learning rate for the optimization.
            alpha (float): Student's t-distribution parameter.
            gamma (float): Weight for the KL term of the T-DPSOM clustering loss.
            beta (float): Weight for the SOM loss.
            kappa (float): Weight for the smoothness loss.
            theta (float): Weight for the VAE loss.
            eta (float): Weight for the prediction loss.
            epochs_pretrain (int): Number of VAE pretraining epochs.
            decay_factor (float): Factor for the learning rate decay.
            name (string): Name of the experiment.
            ex_name (string): Unique name of this particular run.
            logdir (path): Directory for the experiment logs.
            modelpath (path): Path for the model checkpoints.
            validation (bool): If "True" validation set is used for evaluation, otherwise test set is used.
            dropout (float): Dropout factor for the feed-forward layers of the VAE.
            prior (float): Weight of the regularization term of the ELBO.
            val_epochs (bool): If "True" clustering results are saved every 10 epochs on default output files.
            more_runs (bool): Indicator whether to run the job once (False) or multiple times (True) outputting mean and
                              variance.
        """
    input_size = 98
    num_epochs = 100
    batch_size = 300
    latent_dim = 50
    som_dim = [16,16]
    learning_rate = 0.001
    alpha = 10.
    beta = 10.
    gamma = 50.
    kappa = 1.
    theta = 1.
    eta = 1.
    epochs_pretrain = 50
    decay_factor = 0.99
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_LSTM_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()),
                                             uuid.uuid4().hex[:5])
    logdir = "../logs/{}".format(ex_name)
    modelpath = "../models/{}/{}".format(ex_name, ex_name)
    validation = False
    dropout = 0.5
    prior = 0.00001
    annealtime = 200
    lstm_dim = 200
    val_epochs = False
    more_runs = False
    save_pretrain = False
    use_saved_pretrain = False
    
    benchmark=False # Benchmark train time per epoch and return
    train_ratio=1.0 # If changed, use a subset of the training data

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

    #TO DOWNLOAD THE DATA FIRST
    hf = h5py.File('../data/eICU_data.csv', 'r')
    #############################################
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
                yield time_series, i
        elif mode == "val":
            for i in range(len(data_val) // batch_size):
                time_series = data_val[i * batch_size: (i + 1) * batch_size]
                time_series_endpoint = endpoints_total_val[i * batch_size: (i + 1) * batch_size]
                yield time_series, time_series_endpoint, i
        else:
            raise ValueError("The mode has to be in {train, val}")


@ex.capture
def train_model(model, data_train, data_val, endpoints_total_val, lr_val, prior_val, num_epochs, batch_size, latent_dim,
                som_dim, learning_rate, epochs_pretrain, ex_name, logdir, modelpath, val_epochs, save_pretrain,
                use_saved_pretrain, benchmark, train_ratio, annealtime, lstm_dim):

    """Trains the T-DPSOM model.
        Params:
            model (T-DPSOM): T-DPSOM model to train.
            data_train (np.array): Training set.
            data_val (np.array): Validation/test set.
            endpoints_total_val (np.array): Validation/test labels.
            lr_val (tf.Tensor): Placeholder for the learning rate value.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for the training.
            latent_dim (int): Dimensionality of the T-DPSOM's latent space.
            som_dim (list): Dimensionality of the self-organizing map.
            learning_rate (float): Learning rate for the optimization.
            epochs_pretrain (int): Number of VAE pretraining epochs.
            ex_name (string): Unique name of this particular run.
            logdir (path): Directory for the experiment logs.
            modelpath (path): Path for the model checkpoints.
            val_epochs (bool): If "True" clustering results are saved every 10 epochs on default output files.
        """

    max_n_step = 72
    epochs = 0
    iterations = 0
    pretrainpath = "../models/pretrain/LSTM"
    len_data_train = len(data_train)
    len_data_val = len(data_val)
    num_batches = len_data_train // batch_size
    train_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="train")
    val_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="val")

    saver = tf.train.Saver(max_to_keep=5)
    summaries = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_losses = []
        test_losses_mean = []
        with LogFileWriter(ex):
            train_writer = tf.summary.FileWriter(logdir + "/train", sess.graph)
            test_writer = tf.summary.FileWriter(logdir + "/test", sess.graph)
        train_step_SOMVAE, train_step_ae, train_step_som, train_step_prob = model.optimize
        x = model.inputs
        p = model.p
        is_training = model.is_training
        graph = tf.get_default_graph()
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        z_e_rec = graph.get_tensor_by_name('reconstruction_e/decoder/z_e:0')
        training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * batch_size, latent_dim)),
                        init_1: np.zeros((2, batch_size, lstm_dim)), z_e_rec: np.zeros((max_n_step * batch_size, latent_dim))}

        pbar = tqdm(total=(num_epochs+epochs_pretrain*3) * (num_batches))

        print("\n**********Starting job {}********* \n".format(ex_name))
        a = np.zeros((batch_size*72, som_dim[0] * som_dim[1]))
        dp = {p: a}
        dp.update(training_dic)

        if benchmark:
            ttime_per_epoch=[]
            ttime_ae_per_epoch=[]
            ttime_som_per_epoch=[]
            ttime_pred_per_epoch=[]

        if use_saved_pretrain:
            print("\n\nUsing Saved Pretraining...\n")
            saver.restore(sess, pretrainpath)
        else:
            print("\n\nAutoencoder Pretraining...\n")
            if benchmark:
                t_begin_all=timeit.default_timer()
            prior = 0
            for epoch in range(epochs_pretrain):
                if epoch > 10:
                    prior = min(prior + (1. / annealtime), 1.)
                if benchmark:
                    t_begin=timeit.default_timer()
                for i in range(num_batches):
                    batch_data, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: learning_rate, prior_val: prior}
                    f_dic.update(dp)
                    train_step_ae.run(feed_dict=f_dic)
                    if i % 100 == 0:
                        batch_val, _, ii = next(val_gen)
                        f_dic = {x: batch_val}
                        f_dic.update(dp)
                        test_loss, summary = sess.run([model.loss_reconstruction_ze, summaries], feed_dict=f_dic)
                        test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                        f_dic = {x: batch_data}
                        f_dic.update(dp)
                        train_loss, summary = sess.run([model.loss_reconstruction_ze, summaries], feed_dict=f_dic)
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                    pbar.update(1)
                if benchmark:
                    t_end=timeit.default_timer()
                    ttime_ae_per_epoch.append(t_end-t_begin)

            if benchmark:
                t_end_all=timeit.default_timer()
                ttime_ae_pretrain=t_end_all-t_begin_all

            print("\n\nSOM initialization...\n")
            if benchmark:
                t_begin_all=timeit.default_timer()

            for epoch in range(epochs_pretrain//3):
                if benchmark:
                    t_begin=timeit.default_timer()
                for i in range(num_batches):
                    batch_data, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: 0.1}
                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)
                    if i % 100 == 0:
                        batch_val, _, ii = next(val_gen)
                        f_dic = {x: batch_val}
                        f_dic.update(dp)
                        test_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                        f_dic = {x: batch_data}
                        f_dic.update(dp)
                        train_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                    pbar.update(1)
                if benchmark:
                    t_end=timeit.default_timer()
                    ttime_som_per_epoch.append(t_end-t_begin)

            for epoch in range(epochs_pretrain//3):
                if benchmark:
                    t_begin=timeit.default_timer()
                for i in range(num_batches):
                    batch_data, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: 0.01}
                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)
                    if i % 100 == 0:
                        batch_val, _, ii = next(val_gen)
                        f_dic = {x: batch_val}
                        f_dic.update(dp)
                        test_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                        f_dic = {x: batch_data}
                        f_dic.update(dp)
                        train_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                    pbar.update(1)
                if benchmark:
                    t_end=timeit.default_timer()
                    ttime_som_per_epoch.append(t_end-t_begin)

            for epoch in range(epochs_pretrain//3):
                if benchmark:
                    t_begin=timeit.default_timer()
                for i in range(num_batches):
                    batch_data, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: 0.01}
                    f_dic.update(dp)
                    train_step_som.run(feed_dict=f_dic)
                    if i % 100 == 0:
                        batch_val, _, ii = next(val_gen)
                        f_dic = {x: batch_val}
                        f_dic.update(dp)
                        test_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                        f_dic = {x: batch_data}
                        f_dic.update(dp)
                        train_loss, summary = sess.run([model.loss_a, summaries], feed_dict=f_dic)
                        train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                    pbar.update(1)
                if benchmark:
                    t_end=timeit.default_timer()
                    ttime_som_per_epoch.append(t_end-t_begin)

            if benchmark:
                t_end_all=timeit.default_timer()
                ttime_som=t_end_all-t_begin_all

            if save_pretrain:
                saver.save(sess, pretrainpath)

        print("\n\nTraining...\n")

        if benchmark:
            t_begin_all=timeit.default_timer()

        prior = 0
        for epoch in range(num_epochs):
            if epoch > 10:
                prior= min(prior + (1./ annealtime), 1.)
            if benchmark:
                t_begin=timeit.default_timer()
            epochs += 1
            print(epochs)
            f_dic = {x: data_train}
            f_dic.update(training_dic)
            q = []
            for t in range(19):
                q.extend(sess.run(model.q, feed_dict={
                         x: data_train[int(len(data_train) / 20) * t: int(len(data_train) / 20) * (t + 1)]}))
            q.extend(sess.run(model.q, feed_dict={x: data_train[int(len(data_train) / 20) * 19:]}))
            q = np.array(q)
            ppt = model.target_distribution(q)
            q = []
            f_dic = {x: data_val}
            f_dic.update(training_dic)
            for t in range(9):
                q.extend(sess.run(model.q, feed_dict={
                         x: data_val[int(len(data_val) / 10) * t: int(len(data_val) / 10) * (t + 1)]}))
            q.extend(sess.run(model.q, feed_dict={x: data_val[int(len(data_val) / 10) * 9:]}))
            q = np.array(q)
            ppv = model.target_distribution(q)

            for i in range(num_batches):
                iterations += 1
                batch_data, ii = next(train_gen)
                ftrain = {p: ppt[ii*batch_size*72: (ii + 1)*batch_size*72]}
                f_dic = {x: batch_data, lr_val: learning_rate, prior_val: prior}
                f_dic.update(ftrain)
                f_dic.update(training_dic)
                train_step_SOMVAE.run(feed_dict=f_dic)
                train_step_prob.run(feed_dict=f_dic)

                batch_val, _, ii = next(val_gen)
                fval = {p: ppv[ii * batch_size*72: (ii + 1)*batch_size*72]}
                f_dic = {x: batch_val}
                f_dic.update(fval)
                f_dic.update(training_dic)
                test_loss, summary = sess.run([model.loss, summaries], feed_dict=f_dic)
                test_losses.append(test_loss)
                if i % 100 == 0:
                    test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    f_dic = {x: batch_data}
                    f_dic.update(ftrain)
                    f_dic.update(training_dic)
                    train_loss, summary = sess.run([model.loss, summaries], feed_dict=f_dic)
                    if math.isnan(train_loss):
                        return None
                    train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                if i % 1000 == 0:
                    test_loss_mean = np.mean(test_losses)
                    test_losses_mean.append(test_loss_mean)
                    test_losses = []

                if len(test_losses_mean) > 0:
                    test_s = test_losses_mean[-1]
                else:
                    test_s = test_losses_mean

                pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_s, refresh=False)
                pbar.update(1)

            if val_epochs==True and epoch % 5 == 0:
                path = "../models/exp/exp"+ str(epoch)+"/LSTM"
                saver.save(sess, path)
                #results = evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs)

            if benchmark:
                t_end=timeit.default_timer()
                ttime_per_epoch.append(t_end-t_begin)

        if benchmark:
            t_end_all=timeit.default_timer()
            ttime_training=t_end_all-t_begin_all

        print("\n\nPrediction Finetuning...\n")
        if benchmark:
            t_begin_all=timeit.default_timer()

        for epoch in range(50):
            if benchmark:
                t_begin=timeit.default_timer()
            for i in range(num_batches):
                batch_data, ii = next(train_gen)
                f_dic = {x: batch_data, lr_val: learning_rate, prior_val: prior}
                f_dic.update(dp)
                train_step_prob.run(feed_dict=f_dic)
                if i % 100 == 0:
                    batch_val, _, ii = next(val_gen)
                    f_dic = {x: batch_val}
                    f_dic.update(dp)
                    test_loss, summary = sess.run([model.loss_prediction, summaries], feed_dict=f_dic)
                    test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    f_dic = {x: batch_data}
                    f_dic.update(dp)
                    train_loss, summary = sess.run([model.loss_prediction, summaries], feed_dict=f_dic)
                    train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_loss, refresh=False)
                pbar.update(1)
                
            if benchmark:
                t_end=timeit.default_timer()
                ttime_pred_per_epoch.append(t_end-t_begin)

        if benchmark:
            t_end_all=timeit.default_timer()
            ttime_pred=t_end_all-t_begin_all

        saver.save(sess, modelpath)
        results = evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs)
        pbar.close()

        if benchmark:
            print("\nNumber of time series in train: {} %, {}".format(train_ratio, len(data_train)))
            print("SOM init time: {:.3f}".format(ttime_som))
            print("SOM init time per epoch: {:.3f}".format(np.mean(ttime_som_per_epoch)))
            print("AE pretrain time: {:.3f}".format(ttime_ae_pretrain))
            print("AE pretrain time per epoch: {:.3f}".format(np.mean(ttime_ae_per_epoch)))
            print("Training time: {:.3f}".format(ttime_training))
            print("Training time per epoch: {:.3f}".format(np.mean(ttime_per_epoch)))
            print("Pred finetuning time: {:.3f}".format(ttime_pred))
            print("Pred finetuning time per epoch: {:.3f}".format(np.mean(ttime_pred_per_epoch)))
            sys.exit(0)

        return results


@ex.capture
def evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs, batch_size, som_dim, learning_rate, alpha, gamma,
                   beta , theta, epochs_pretrain, ex_name, kappa, dropout, prior, latent_dim, eta, lstm_dim):
    """Evaluates the performance of the trained model in terms of normalized
        mutual information adjusted mutual information score and purity.

        Args:
            model (T-DPSOM): Trained T-DPSOM model to evaluate.
            x (tf.Tensor): Input tensor or placeholder.
            val_gen (generator): Val/Test generator for the batches.
            len_data_val (int): Length of validation set.
            modelpath (path): Path from which to restore the model.
            epochs (int): number of epochs of training.
            batch_size (int): Batch size for the training.
            som_dim (list): Dimensionality of the self-organizing map.
            learning_rate (float): Learning rate for the optimization.
            alpha (float): Student's t-distribution parameter.
            gamma (float): Weight for the KL term of the PSOM clustering loss.
            beta (float): Weight for the SOM loss.
            theta (float): Weight for the VAE loss.
            epochs_pretrain (int): Number of VAE pretraining epochs.
            ex_name (string): Unique name of this particular run.
            kappa (float): Weight for the smoothness loss.
            dropout (float): Dropout factor for the feed-forward layers of the VAE.
            prior (float): Weight of the regularization term of the ELBO.
            latent_dim (int): Dimensionality of the T-DPSOM's latent space.
            eta (float): Weight for the prediction loss.

        Returns:
            dict: Dictionary of evaluation results (NMI, AMI, Purity).
        """

    max_n_step = 72 #length of the time-series

    saver = tf.train.Saver(keep_checkpoint_every_n_hours=2.)
    num_batches = len_data_val // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)

        is_training = model.is_training
        graph = tf.get_default_graph()
        init_1 = graph.get_tensor_by_name("prediction/next_state/init_state:0")
        z_e_p = graph.get_tensor_by_name("prediction/next_state/input_lstm:0")
        z_e_rec = graph.get_tensor_by_name('reconstruction_e/decoder/z_e:0')
        training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * batch_size, latent_dim)),
                        init_1: np.zeros((2, batch_size, lstm_dim)), z_e_rec: np.zeros((max_n_step * batch_size, latent_dim))}

        test_k_all = []
        labels_val_all = []
        z_q_all = []
        z_e_all = []
        print("Evaluation...")

        for i in range(num_batches):
            batch_data, batch_labels, ii = next(val_gen)
            f_dic = {x: batch_data}
            f_dic.update(training_dic)
            test_k_all.extend(sess.run(model.k, feed_dict=f_dic))
            labels_val_all.extend(batch_labels)
            z_q_all.extend(sess.run(model.z_q, feed_dict=f_dic))
            z_e_all.extend(sess.run(model.z_e_sample, feed_dict=f_dic))

        labels_val_all = np.array(labels_val_all)
        test_k_all = np.array(test_k_all)
        labels_val_all = np.reshape(labels_val_all, (-1, labels_val_all.shape[-1]))
        print("Mean: {:.3f}, Std: {:.3f}".format(np.mean(labels_val_all[:,3]), np.std(labels_val_all[:,3])))
        NMI_24 = metrics.normalized_mutual_info_score(labels_val_all[:, 3], test_k_all, average_method='geometric')
        NMI_12 = metrics.normalized_mutual_info_score(labels_val_all[:, 2], test_k_all, average_method='geometric')
        NMI_6 = metrics.normalized_mutual_info_score(labels_val_all[:, 1], test_k_all, average_method='geometric')
        NMI_1 = metrics.normalized_mutual_info_score(labels_val_all[:, 0], test_k_all, average_method='geometric')
        AMI_1 = metrics.adjusted_mutual_info_score(test_k_all, labels_val_all[:, 0])

        mean = np.sum(labels_val_all[:, 0]) / len(labels_val_all[:, 0])
        ones = np.ones((len(np.reshape(test_k_all, (-1)))))
        clust_matr1 = np.zeros(som_dim[0] * som_dim[1])
        labels = labels_val_all[:, 0]
        for i in range(som_dim[0] * som_dim[1]):
            dd = np.sum(ones[np.where(np.reshape(test_k_all, (-1)) == i)])
            if dd == 0:
                s1 = 0
            else:
                s1 = np.sum(labels[np.where(np.reshape(test_k_all, (-1)) == i)]) / np.sum(
                    ones[np.where(np.reshape(test_k_all, (-1)) == i)])
            clust_matr1[i] = s1

        sd = som_dim[0]*som_dim[1]
        k = np.arange(0, sd)
        k1 = k // som_dim[0]
        k2 = k % som_dim[1]
        W = np.zeros((sd, sd))
        for i in range(sd):
            for j in range(sd):
                d1 = np.abs((k1[i] - k1[j]))
                d2 = np.abs((k2[i] - k2[j]))
                d1 = min(som_dim[0] - d1, d1)
                d2 = min(som_dim[1] - d2, d2)
                W[i, j] = np.exp(-(d1 + d2))

        M = 0
        N_n = 0
        for i in range(sd):
            for j in range(sd):
                M += (clust_matr1[i] - mean) * (clust_matr1[j] - mean) * W[i, j]
        for i in range(sd):
            N_n += (clust_matr1[i] - mean) ** 2
        W_n = np.sum(W)
        I = M * sd / (N_n * W_n)

    results = {}
    results["NMI_24"] = NMI_24
    results["NMI_12"] = NMI_12
    results["NMI_6"] = NMI_6
    results["NMI_1"] = NMI_1
    results["AMI_1"] = AMI_1
    results["MI"] = I

    f = open("results_eICU.txt", "a+")
    f.write("Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, "
            "theta= %f, eta= %f, beta= %f, alpha=%f, gamma=%f, epochs_pretrain=%d, dropout= %f, prior= %f"
                % (epochs, som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, eta, beta,
                   alpha, gamma, epochs_pretrain, dropout, prior))
    f.write(", kappa= %f, NMI24: %f, NMI12: %f, NMI6: %f, NMI1: %f, AMI1: %f, I: %f.Name: %r \n"
            % (kappa, results["NMI_24"], results["NMI_12"], results["NMI_6"], results["NMI_1"], results["AMI_1"],
               results["MI"], ex_name))
    f.close()

    return results

@ex.capture
def z_dist_flat(z_e, embeddings, som_dim, latent_dim):
    """Computes the distances between the encodings and the embeddings."""
    emb = np.reshape(embeddings, (som_dim[0]*som_dim[1], -1))
    z = np.reshape(z_e, (z_e.shape[0], 1, latent_dim))
    z = np.tile(z, [1,som_dim[0]*som_dim[1], 1])
    z_dist = np.square(z-emb)
    z_dist_red = np.sum(z_dist, axis=-1)
    return z_dist_red


@ex.automain
def main(input_size, latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, theta, ex_name, kappa, prior,
         more_runs, dropout, eta, epochs_pretrain, batch_size, num_epochs, train_ratio, annealtime, modelpath, lstm_dim):

    input_channels = 98

    lr_val = tf.placeholder_with_default(learning_rate, [])
    prior_val = tf.placeholder_with_default(prior, [])

    model = TDPSOM(input_size=input_size, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val,
                   decay_factor=decay_factor, dropout=dropout, input_channels=input_channels, alpha=alpha, beta=beta,
                   eta=eta, kappa=kappa, theta=theta, gamma=gamma, prior=prior, lstm_dim=lstm_dim)

    data_train, data_val, _, endpoints_total_val = get_data()

    if train_ratio<1.0:
        data_train=data_train[:int(len(data_train)*train_ratio)]

    if not more_runs:
        results = train_model(model, data_train, data_val, endpoints_total_val, lr_val, prior_val)

#################################################################################################################################################

        tf.reset_default_graph()
        val_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="val")
        num_batches = len(data_val) // 300
        num_pred = 6
        som = 16 * 16
        max_n_step = 72
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.import_meta_graph(modelpath + ".meta")
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
            z_p = graph.get_tensor_by_name('reconstruction_e/decoder/z_e:0')
            reconstruction = graph.get_tensor_by_name("reconstruction_e/x_hat:0")

            print("Evaluation...")
            training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * len(data_val), latent_dim)),
                            init_1: np.zeros((2, batch_size, lstm_dim)),
                            z_p: np.zeros((max_n_step * len(data_val), latent_dim))}
            k_all = []
            z_e_all = []
            z_q_all = []
            qq = []
            x_rec = []
            for i in range(num_batches):
                batch_data, batch_labels, ii = next(val_gen)
                f_dic = {x: batch_data}
                k_all.extend(sess.run(k, feed_dict=f_dic))
                z_q_all.extend(sess.run(q, feed_dict=f_dic))
                z_e_all.extend(sess.run(z_e, feed_dict=f_dic))
                qq.extend(sess.run(q, feed_dict=f_dic))
                f_dic.update(training_dic)
                x_rec.extend(sess.run(reconstruction, feed_dict=f_dic))
            z_e_all = np.array(z_e_all)
            k_all = np.array(k_all)
            qq = np.array(qq)
            x_rec = np.array(x_rec)
            z_e_all = z_e_all.reshape((-1, max_n_step, latent_dim))
            k_all = k_all.reshape((-1, max_n_step))

            t = 72 - num_pred

            embeddings = sess.run(embeddings, feed_dict={x: data_val[:, :t, :]})
            embeddings = np.reshape(embeddings, (-1, latent_dim))

            z_e_o = z_e_all[:, :t, :]
            k_o = k_all[:, :t]
            k_eval = []
            next_z_e_o = []
            state1_o = []
            for i in range(num_batches):
                batch_data, batch_labels, ii = next(val_gen)
                batch_data = batch_data[:, :t, :]
                f_dic = {x: batch_data}
                f_dic.update(training_dic)
                next_z_e_o.extend(sess.run(next_z_e, feed_dict=f_dic))
                if i == 0:
                    state1_o = sess.run(state1, feed_dict=f_dic)
                else:
                    state1_o = np.concatenate([state1_o, sess.run(state1, feed_dict=f_dic)], axis=1)
            next_z_e_o = np.array(next_z_e_o)
            state1_o = np.array(state1_o)

            next_z_e_o_all = np.reshape(next_z_e_o[:, -1, :], (-1, 1, latent_dim))
            next_z_e_o = next_z_e_o[:, -1, :]
            k_next = np.argmin(z_dist_flat(next_z_e_o, embeddings), axis=-1)
            k_o = np.concatenate([k_o, np.expand_dims(k_next, 1)], axis=1)
            z_e_o = np.concatenate([z_e_o, np.expand_dims(next_z_e_o, 1)], axis=1)
            f_dic = {x: np.zeros((len(data_val), 1, 98)), is_training: False,
                     z_e_p: np.zeros((1 * len(data_val), latent_dim)),
                     z_p: next_z_e_o, init_1: np.zeros((2, batch_size, lstm_dim))}
            x_pred_hat = np.reshape(sess.run(reconstruction, feed_dict=f_dic), (-1, 1, 98))

            for i in range(num_pred - 1):
                print(i)
                inp = data_val[:1500, (t + i), :]
                f_dic = {x: np.reshape(inp, (inp.shape[0], 1, inp.shape[1]))}
                val_dic = {is_training: False, z_e_p: next_z_e_o, init_1: state1_o,
                           z_p: np.zeros((max_n_step * len(inp), latent_dim))}
                f_dic.update(val_dic)
                next_z_e_o = sess.run(next_z_e, feed_dict=f_dic)
                state1_o = sess.run(state1, feed_dict=f_dic)
                next_z_e_o_all = np.concatenate([next_z_e_o_all, next_z_e_o], axis=1)
                k_next = np.argmin(z_dist_flat(next_z_e_o, embeddings), axis=-1)
                k_o = np.concatenate([k_o, np.expand_dims(k_next, 1)], axis=1)
                z_e_o = np.concatenate([z_e_o, next_z_e_o], axis=1)
                next_z_e_o = np.reshape(next_z_e_o, (-1, latent_dim))
                f_dic = {x: np.zeros((len(data_val), 1, 98)), is_training: False,
                         z_e_p: np.zeros((max_n_step * len(data_val), latent_dim)),
                         z_p: next_z_e_o, init_1: np.zeros((2, batch_size, lstm_dim))}
                final_x = sess.run(reconstruction, feed_dict=f_dic)
                x_pred_hat = np.concatenate([x_pred_hat, np.reshape(final_x, (-1, 1, 98))], axis=1)

            f_dic = {x: np.zeros((1500, 1, 98)), is_training: False, z_e_p: np.zeros((max_n_step * 1500, latent_dim)),
                     z_p: z_e_all[:, t - 1, :], init_1: np.zeros((2, batch_size, lstm_dim))}
            final_x = sess.run(reconstruction, feed_dict=f_dic)

        pred_ze = sklearn.metrics.mean_squared_error(np.reshape(next_z_e_o_all[:, :], (-1, latent_dim)),
                                                     np.reshape(z_e_all[:, -num_pred:], (-1, latent_dim)))
        pred_rec = sklearn.metrics.mean_squared_error(np.reshape(x_rec, (-1, 98)),
                                                      np.reshape(data_val[:1500, :], (-1, 98)))
        pred_xhat = sklearn.metrics.mean_squared_error(np.reshape(x_pred_hat, (-1, 98)),
                                                       np.reshape(data_val[:1500, -num_pred:], (-1, 98)))

        f = open("results_eICU_pred.txt", "a+")
        f.write("Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, "
                "theta= %f, eta= %f, beta= %f, alpha=%f, gamma=%f, epochs_pretrain=%d, dropout= %f, annealtime= %d, "
                % (num_epochs, som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, eta, beta,
                   alpha, gamma, epochs_pretrain, dropout, annealtime))
        f.write(", kappa= %f, pred_ze: %f, pred_rec: %f, pred_xhat: %f.Name: %r \n"
                % (kappa, pred_ze, pred_rec, pred_xhat, ex_name))
        f.close()

#################################################################################################################################################

    else:
        NMI_24_all=[]
        NMI_12_all=[]
        NMI_6_all = []
        NMI_1_all = []
        MI_all = []
        T = 10
        for i in range(T):
            results = train_model(model, data_train, data_val, endpoints_total_val, lr_val, prior_val)
            if results is None:
                T += 1
                if T > 15:
                    f = open("evaluation_eICU.txt", "a+")
                    f.write(
                        "som_dim=[%d,%d], latent_dim= %d, batch_size= %d, "
                        "learning_rate= %f, theta= %f, dropout=%f, prior=%f, kappa=%d, gamma=%d, beta%d, eta=%f, "
                        "epochs_pretrain=%d, epochs= %d, NOT WORKING !!\n"
                        % (som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate,
                        theta, dropout, prior, kappa, gamma, beta, eta, epochs_pretrain, num_epochs))
                    return 0
            else:
                NMI_24_all.append(results["NMI_24"])
                NMI_12_all.append(results["NMI_12"])
                NMI_6_all.append(results["NMI_6"])
                NMI_1_all.append(results["NMI_1"])
                MI_all.append(results["MI"])

        NMI_24_mean = np.mean(NMI_24_all)
        NMI_24_sd = np.std(NMI_24_all)  / np.sqrt(10)
        NMI_12_mean = np.mean(NMI_12_all)
        NMI_12_sd = np.std(NMI_12_all) / np.sqrt(10)
        NMI_6_mean = np.mean(NMI_6_all)
        NMI_6_sd = np.std(NMI_6_all) / np.sqrt(10)
        NMI_1_mean = np.mean(NMI_1_all)
        NMI_1_sd = np.std(NMI_1_all) / np.sqrt(10)
        MI_mean = np.mean(MI_all)
        MI_sd = np.std(MI_all) / np.sqrt(10)

        f = open("evaluation_eICU.txt", "a+")
        f.write(
                "som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, theta= %f, "
                "dropout=%f, prior=%f, kappa=%d, gamma=%d, beta%d, eta=%f, epochs_pretrain=%d, epochs= %d"
                % (som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, dropout, prior,
                   kappa, gamma, beta, eta, epochs_pretrain, num_epochs))


        f.write(", T= %d, RESULTS NMI24: %f + %f, NMI12: %f + %f, NMI6: %f + %f, NMI1: %f + %f, MI: %f + %f.  Name: %r \n"
                % (T, NMI_24_mean, NMI_24_sd, NMI_12_mean, NMI_12_sd, NMI_6_mean, NMI_6_sd, NMI_1_mean, NMI_1_sd,
                   MI_mean, MI_sd, ex_name))
        f.close()

    return results
