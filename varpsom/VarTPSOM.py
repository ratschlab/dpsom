"""
Script for training the VarTPSOM model
"""

import uuid
from datetime import date
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import sacred
from sacred.stflow import LogFileWriter
import math
import h5py
from sklearn import metrics
from VarTPSOM_model import VarTPSOM
from sklearn.model_selection import train_test_split

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
            latent_dim (int): Dimensionality of the VarTPSOM's latent space.
            som_dim (list): Dimensionality of the self-organizing map.
            learning_rate (float): Learning rate for the optimization.
            alpha (float): Student's t-distribution parameter.
            gamma (float): Weight for the KL term of the VarTPSOM clustering loss.
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
    num_epochs = 50
    batch_size = 300
    latent_dim = 10
    som_dim = [8, 8]
    learning_rate = 0.001
    alpha = 10.
    beta = 50.
    gamma = 100.
    kappa = 25.
    theta = 0.001
    eta = 1.
    epochs_pretrain = 30
    decay_factor = 0.99
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_LSTM_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()),
                                             uuid.uuid4().hex[:5])
    logdir = "../logs/{}".format(ex_name)
    modelpath = "../models/{}/{}".format(ex_name, ex_name)
    validation = False
    dropout = 0.5
    prior = 0.001
    val_epochs = False
    more_runs = False


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

    hf = h5py.File('../data/eICU_data.csv', 'r')
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
def train_model(model, data_train, data_val, endpoints_total_val, lr_val, num_epochs, batch_size, latent_dim, som_dim,
                learning_rate, epochs_pretrain, ex_name, logdir, modelpath, val_epochs):

    """Trains the VarTPSOM model.
        Params:
            model (VarPSOM): VarPSOM model to train.
            data_train (np.array): Training set.
            data_val (np.array): Validation/test set.
            endpoints_total_val (np.array): Validation/test labels.
            lr_val (tf.Tensor): Placeholder for the learning rate value.
            num_epochs (int): Number of training epochs.
            batch_size (int): Batch size for the training.
            latent_dim (int): Dimensionality of the VarTPSOM's latent space.
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
    len_data_train = len(data_train)
    len_data_val = len(data_val)
    num_batches = len_data_train // batch_size
    train_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="train")
    val_gen = batch_generator(data_train, data_val, endpoints_total_val, mode="val")

    saver = tf.train.Saver(max_to_keep=2)
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
        training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * batch_size, latent_dim)),
                        init_1: np.zeros((2, batch_size, 10))}

        pbar = tqdm(total=(num_epochs+epochs_pretrain*3) * (num_batches))

        print("\n**********Starting job {}********* \n".format(ex_name))
        a = np.zeros((batch_size*72, som_dim[0] * som_dim[1]))
        dp = {p: a}
        dp.update(training_dic)

        print("\n\nAutoencoder Pretraining...\n")
        for epoch in range(epochs_pretrain):
            for i in range(num_batches):
                batch_data, ii = next(train_gen)
                f_dic = {x: batch_data, lr_val: learning_rate}
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

        print("\n\nSOM initialization...\n")
        for epoch in range(epochs_pretrain//3):
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
        for epoch in range(epochs_pretrain//3):
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

        for epoch in range(epochs_pretrain//3):
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

        print("\n\nTraining...\n")
        for epoch in range(num_epochs):
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
                f_dic = {x: batch_data, lr_val: learning_rate}
                f_dic.update(ftrain)
                f_dic.update(training_dic)
                train_step_SOMVAE.run(feed_dict=f_dic)
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

            if val_epochs==True:
                saver.save(sess, modelpath)
                results = evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs)

        saver.save(sess, modelpath)
        results = evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs)
        pbar.close()
        return results


@ex.capture
def evaluate_model(model, x, val_gen, len_data_val, modelpath, epochs, batch_size, som_dim, learning_rate, alpha, gamma,
                   beta , theta, epochs_pretrain, ex_name, kappa, dropout, prior, latent_dim, eta):
    """Evaluates the performance of the trained model in terms of normalized
        mutual information adjusted mutual information score and purity.

        Args:
            model (VarPSOM): Trained VarPSOM model to evaluate.
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
            latent_dim (int): Dimensionality of the VarTPSOM's latent space.
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
        training_dic = {is_training: True, z_e_p: np.zeros((max_n_step * batch_size, latent_dim)),
                        init_1: np.zeros((2, batch_size, 10))}

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
        labels_val_all = np.reshape(labels_val_all, (-1, 8))
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
    results["AMI_1"] = AMI_1

    f = open("results_eICU.txt", "a+")
    f.write("Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, "
            "theta= %f, eta= %f, beta= %f, alpha=%f, gamma=%f, epochs_pretrain=%d, dropout= %f, prior= %f"
                % (epochs, som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, eta, beta,
                   alpha, gamma, epochs_pretrain, dropout, prior))
    f.write(", kappa= %f, NMI24: %f, NMI12: %f, NMI6: %f, NMI1: %f, AMI1: %f.Name: %r \n"
            % (kappa, results["NMI_24"], results["NMI_12"], results["NMI_6"], results["NMI_1"], results["AMI_1"],
               ex_name))
    f.close()

    return results


@ex.automain
def main(input_size, latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, theta, ex_name, kappa, prior,
         more_runs, dropout, eta, epochs_pretrain, batch_size, num_epochs):

    input_channels = 98

    lr_val = tf.placeholder_with_default(learning_rate, [])

    model = VarTPSOM(input_size=input_size, latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val,
                     decay_factor=decay_factor, dropout=dropout, input_channels=input_channels, alpha=alpha, beta=beta,
                     eta=eta, kappa=kappa, theta=theta, gamma=gamma, prior=prior)

    data_train, data_val, _, endpoints_total_val = get_data()

    if not more_runs:
        results = train_model(model, data_train, data_val, endpoints_total_val, lr_val)

    else:
        NMI_24_all=[]
        NMI_12_all=[]
        NMI_6_all = []
        NMI_1_all = []
        T = 10
        for i in range(T):
            results = train_model(model, data_train, data_val, endpoints_total_val, lr_val)
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

        NMI_24_mean = np.mean(NMI_24_all)
        NMI_24_sd = np.std(NMI_24_all)
        NMI_12_mean = np.mean(NMI_12_all)
        NMI_12_sd = np.std(NMI_12_all)
        NMI_6_mean = np.mean(NMI_6_all)
        NMI_6_sd = np.std(NMI_6_all)
        NMI_1_mean = np.mean(NMI_1_all)
        NMI_1_sd = np.std(NMI_1_all)

        f = open("evaluation_eICU.txt", "a+")
        f.write(
                "som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, theta= %f, "
                "dropout=%f, prior=%f, kappa=%d, gamma=%d, beta%d, eta=%f, epochs_pretrain=%d, epochs= %d"
                % (som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, dropout, prior,
                   kappa, gamma, beta, eta, epochs_pretrain, num_epochs))


        f.write(", T= %d, RESULTS NMI24: %f + %f, NMI12: %f + %f, NMI6: %f + %f, NMI1: %f + %f.  Name: %r \n"
                % (T, NMI_24_mean, NMI_24_sd, NMI_12_mean, NMI_12_sd, NMI_6_mean, NMI_6_sd, NMI_1_mean, NMI_1_sd,
                   ex_name))
        f.close()

    return results
