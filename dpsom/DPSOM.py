"""
Script to train the DPSOM model.
"""

import uuid
from datetime import date
from pathlib import Path

try:
    import tensorflow.compat.v1 as tf 
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

from tqdm import tqdm
import sacred
from sacred.stflow import LogFileWriter
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import metrics
from DPSOM_model import DPSOM
from utils import cluster_purity
import os
from os import path
import time
import random
import csv
import numpy.random as nprand

ex = sacred.Experiment("hyperopt")
ex.observers.append(sacred.observers.FileStorageObserver.create("../sacred_runs"))
ex.captured_out_filter = sacred.utils.apply_backspaces_and_linefeeds

@ex.config
def ex_config():
    """Sacred configuration for the experiment.
    Params:
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the DPSOM's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        alpha (float): Student's t-distribution parameter.
        gamma (float): Weight for the KL term of the PSOM clustering loss.
        beta (float): Weight for the SOM loss.
        theta (float): Weight for the VAE loss.
        epochs_pretrain (int): Number of VAE pretraining epochs.
        decay_factor (float): Factor for the learning rate decay.
        decay_steps (float): Number of steps for the learning rate decay.
        name (string): Name of the experiment.
        ex_name (string): Unique name of this particular run.
        logdir (path): Directory for the experiment logs.
        modelpath (path): Path for the model checkpoints.
        data_set (string): Data set for the training.
        validation (bool): If "True" validation set is used for evaluation, otherwise test set is used.
        dropout (float): Dropout factor for the feed-forward layers of the VAE.
        prior_var (float): Multiplier of the diagonal variance of the VAE multivariate gaussian prior.
        prior (float): Weight of the regularization term of the ELBO.
        convolution (bool): Indicator if the model use convolutional layers (True) or feed-forward layers (False).
        val_epochs (bool): If "True" clustering results are saved every 10 epochs on default output files.
        more_runs (bool): Indicator whether to run the job once (False) or multiple times (True) outputting mean and
                          variance.
    """
    num_epochs = 300
    batch_size = 300
    latent_dim = 100
    som_dim = [8, 8]
    learning_rate = 0.001
    learning_rate_pretrain = 0.001
    alpha = 10.0
    beta = 0.25
    gamma = 20
    theta = 1
    epochs_pretrain = 15
    decay_factor = 0.99
    decay_steps = 5000
    name = ex.get_experiment_info()["name"]
    ex_name = "{}_{}_{}-{}_{}_{}".format(name, latent_dim, som_dim[0], som_dim[1], str(date.today()),
                                         uuid.uuid4().hex[:5])
    logdir = "../logs/{}".format(ex_name)
    modelpath = "../models/{}/{}.ckpt".format(ex_name, ex_name)
    data_set = "MNIST"
    validation = False
    dropout = 0.4
    prior_var = 1
    prior = 0.5
    convolution = False
    val_epochs = False
    more_runs = False
    use_saved_pretrain = False
    save_pretrain = False
    random_seed=2020

    exp_output=False # Output to Google Cloud File System
    exp_path="/home/mhueser/data/variational_psom/static_clustering_FMNIST/robustness"

@ex.capture
def get_data_generator(data_train, data_val, labels_train, labels_val, data_test, labels_test):
    """Creates a data generator for the training.
    Args:
        data_train: training set.
        data_val: validation set.
        labels_train: labels of the training set.
        labels_val: labels of the validation set.
        data_test: test set.
        labels_test: labels of the test set.

    Returns:
        generator: Data generator for the batches."""


    def batch_generator(mode="train", batch_size=300):
        """Generator for the data batches.

        Args:
            mode (str): Mode in ['train', 'val', 'test'] that decides which data set the generator
                samples from (default: 'train').
            batch_size (int): The size of the batches (default: 300).

        Yields:
            np.array: Data batch.
            np.array: Labels batch.
            int: Offset of the batch in dataset.
        """
        assert mode in ["train", "val", "test"], "The mode should be in {train, val, test}."
        if mode == "train":
            images = data_train.copy()
            labels = labels_train.copy()
        elif mode == "val":
            images = data_val.copy()
            labels = labels_val.copy()
        elif mode == "test":
            images = data_test.copy()
            labels = labels_test.copy()

        while True:
            for i in range(len(images) // batch_size):
                yield images[i * batch_size:(i + 1) * batch_size], labels[i * batch_size:(i + 1) * batch_size], i

    return batch_generator


@ex.capture
def train_model(model, data_train, data_val, generator, lr_val, num_epochs, batch_size, logdir, ex_name, validation,
                val_epochs, modelpath, learning_rate, epochs_pretrain, som_dim, latent_dim, use_saved_pretrain,
                learning_rate_pretrain, save_pretrain):

    """Trains the DPSOM model.
    Args:
        model (DPSOM): DPSOM model to train.
        data_train (np.array): Training set.
        data_val (np.array): Validation/test set.
        generator (generator): Data generator for the batches.
        lr_val (tf.Tensor): Placeholder for the learning rate value.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for the training.
        logdir (path): Directory for the experiment logs.
        ex_name (string): Unique name of this particular run.
        val_epochs (bool): If "True" clustering results are saved every 10 epochs on default output files.
        modelpath (path): Path for the model checkpoints.
        learning_rate (float): Learning rate for the optimization.
        epochs_pretrain (int): Number of VAE pretraining epochs.
        som_dim (list): Dimensionality of the self-organizing map.
        latent_dim (int): Dimensionality of the DPSOM's latent space.
    """
    epochs = 0
    iterations = 0
    train_gen = generator("train", batch_size)
    if validation:
        val_gen = generator("val", batch_size)
    else:
        val_gen = generator("test", batch_size)
    len_data_train = len(data_train)
    len_data_val = len(data_val)
    num_batches = len_data_train // batch_size

    saver = tf.train.Saver(max_to_keep=5)
    summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        test_losses = []
        test_losses_mean = []
        pretrainpath = "../models/pretrainVAE/VAE"
        with LogFileWriter(ex):
            train_writer = tf.summary.FileWriter(logdir + "/train", sess.graph)
            test_writer = tf.summary.FileWriter(logdir + "/test", sess.graph)
        train_step_VARPSOM, train_step_vae, train_step_som = model.optimize
        x = model.inputs
        p = model.p
        is_training = model.is_training
        graph = tf.get_default_graph()
        z = graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")

        print("\n**********Starting job {}********* \n".format(ex_name))
        pbar = tqdm(total=(num_epochs + epochs_pretrain + 40) * num_batches)

        if use_saved_pretrain:
            print("\n\nUsing Saved Pretraining...\n")
            saver.restore(sess, pretrainpath)
        else:
            print("\n\nAutoencoder Pretraining...\n")
            a = np.zeros((batch_size, som_dim[0] * som_dim[1]))
            dp = {p: a, is_training: True, z: np.zeros((batch_size, latent_dim))}
            for epoch in range(epochs_pretrain):
                for i in range(num_batches):
                    batch_data, _, _ = next(train_gen)
                    f_dic = {x: batch_data, lr_val: learning_rate_pretrain}
                    f_dic.update(dp)
                    train_step_vae.run(feed_dict=f_dic)
                    if i % 100 == 0:
                        batch_val, _, _ = next(val_gen)
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
            for epoch in range(5):
                for i in range(num_batches):
                    batch_data, _, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: 0.9}
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
            for epoch in range(5):
                for i in range(num_batches):
                    batch_data, _, ii = next(train_gen)
                    f_dic = {x: batch_data, lr_val: 0.3}
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
            for epoch in range(5):
                for i in range(num_batches):
                    batch_data, _, ii = next(train_gen)
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
            for epoch in range(5):
                for i in range(num_batches):
                    batch_data, _, ii = next(train_gen)
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
            if save_pretrain:
                saver.save(sess, pretrainpath)

        print("\n\nTraining...\n")
        lratios=[]
        l2ratios=[]
        l3ratios=[]
        for epoch in range(num_epochs):
            epochs += 1
            #Compute initial soft probabilities between data points and centroids
            q = []
            for t in range(9):
                q.extend(sess.run(model.q, feed_dict={
                        x: data_train[int(len(data_train) / 10) * t: int(len(data_train) / 10) * (t + 1)],
                        is_training: True, z: np.zeros((int(len(data_train) / 10), latent_dim))}))
            q.extend(sess.run(model.q, feed_dict={x: data_train[int(len(data_train) / 10) * 9:], is_training: True,
                                                      z: np.zeros((int(len(data_train) / 10), latent_dim))}))
            q = np.array(q)
            ppt = model.target_distribution(q)
            q = sess.run(model.q, feed_dict={x: data_val, is_training: True, z: np.zeros((len(data_val), latent_dim))})
            ppv = model.target_distribution(q)

            #Train
            for i in range(num_batches):
                iterations += 1
                batch_data, _, ii = next(train_gen)
                ftrain = {p: ppt[ii * batch_size: (ii + 1) * batch_size], is_training: True,
                              z: np.zeros((batch_size, latent_dim))}
                f_dic = {x: batch_data, lr_val: learning_rate}
                f_dic.update(ftrain)
                train_step_VARPSOM.run(feed_dict=f_dic)
                batch_val, _, ii = next(val_gen)
                fval = {p: ppv[ii * batch_size: (ii + 1) * batch_size], is_training: True,
                            z: np.zeros((batch_size, latent_dim))}
                f_dic = {x: batch_val}
                f_dic.update(fval)
                test_loss, summary = sess.run([model.loss, summaries], feed_dict=f_dic)
                test_losses.append(test_loss)
                if i % 100 == 0:
                    test_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                    f_dic = {x: batch_data}
                    f_dic.update(ftrain)
                    train_loss, summary = sess.run([model.loss, summaries], feed_dict=f_dic)
                    elbo_loss=sess.run([model.theta*model.loss_reconstruction_ze], feed_dict=f_dic)
                    cah_loss=sess.run([model.gamma*model.loss_commit], feed_dict=f_dic)
                    ssom_loss=sess.run([model.beta*model.loss_som], feed_dict=f_dic)
                    cah_ssom_ratio=cah_loss[0]/ssom_loss[0]
                    vae_cah_ratio=elbo_loss[0]/cah_loss[0]
                    clust_vae_ratio=elbo_loss[0]/(ssom_loss[0]+cah_loss[0])
                    lratios.append(cah_ssom_ratio)
                    l2ratios.append(vae_cah_ratio)
                    l3ratios.append(clust_vae_ratio)
                    train_writer.add_summary(summary, tf.train.global_step(sess, model.global_step))
                if i % 1000 == 0:
                    test_loss_mean = np.mean(test_losses)
                    test_losses_mean.append(test_loss_mean)
                    test_losses = []

                if len(test_losses_mean) > 0:
                    test_s = test_losses_mean[-1]
                else:
                    test_s = test_losses_mean

                pbar.set_postfix(epoch=epoch, train_loss=train_loss, test_loss=test_s,
                                 ssom=ssom_loss, cah=cah_loss, vae=elbo_loss, cs_ratio=np.mean(lratios),
                                 vc_ratio=np.mean(l2ratios), cr_ratio=np.mean(l3ratios), refresh=False)
                pbar.update(1)
            saver.save(sess, modelpath)

            if val_epochs == True and epochs % 10 == 0:
                saver.save(sess, modelpath)
                results = evaluate_model(model, generator, len_data_val, x, modelpath, epochs)
                if results is None:
                    return None

        saver.save(sess, modelpath)
        results = evaluate_model(model, generator, len_data_val, x, modelpath, epochs)
    return results


@ex.capture
def evaluate_model(model, generator, len_data_val, x, modelpath, epochs, batch_size, latent_dim, som_dim,
                   learning_rate, alpha, gamma, beta, theta, epochs_pretrain, decay_factor, ex_name, data_set,
                   validation, dropout, prior_var, prior, convolution):

    """Evaluates the performance of the trained model in terms of normalized
    mutual information adjusted mutual information score and purity.

    Args:
        model (DPSOM): Trained DPSOM model to evaluate.
        generator (generator): Data generator for the batches.
        len_data_val (int): Length of validation set.
        x (tf.Tensor): Input tensor or placeholder.
        modelpath (path): Path from which to restore the model.
        epochs (int): number of epochs of training.
        batch_size (int): Batch size for the training.
        latent_dim (int): Dimensionality of the DPSOM's latent space.
        som_dim (list): Dimensionality of the self-organizing map.
        learning_rate (float): Learning rate for the optimization.
        alpha (float): Student's t-distribution parameter.
        gamma (float): Weight for the KL term of the PSOM clustering loss.
        beta (float): Weight for the SOM loss.
        theta (float): Weight for the VAE loss.
        epochs_pretrain (int): Number of VAE pretraining epochs.
        decay_factor (float): Factor for the learning rate decay.
        ex_name (string): Unique name of this particular run.
        data_set (string): Data set for the training.
        validation (bool): If "True" validation set is used for evaluation, otherwise test set is used.
        dropout (float): Dropout factor for the feed-forward layers of the VAE.
        prior_var (float): Multiplier of the diagonal variance of the VAE multivariate gaussian prior.
        prior (float): Weight of the regularization term of the ELBO.
        convolution (bool): Indicator if the model use convolutional layers (True) or feed-forward layers (False).

    Returns:
        dict: Dictionary of evaluation results (NMI, AMI, Purity).
    """
    saver = tf.train.Saver()
    num_batches = len_data_val // batch_size

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, modelpath)
        graph = tf.get_default_graph()
        z = graph.get_tensor_by_name("reconstruction_e/decoder/z_e:0")
        is_training = model.is_training

        if validation:
            val_gen = generator("val", batch_size)
        else:
            val_gen = generator("test", batch_size)

        test_k_all = []
        labels_val_all = []
        print("Evaluation...")
        for i in range(num_batches):
            batch_data, batch_labels, ii = next(val_gen)
            labels_val_all.extend(batch_labels)
            test_k = sess.run(model.k,
                                  feed_dict={x: batch_data, is_training: True, z: np.zeros((batch_size, latent_dim))})

            test_k_all.extend(test_k)

        test_nmi = metrics.normalized_mutual_info_score(np.array(labels_val_all), test_k_all, average_method='geometric')
        test_purity = cluster_purity(np.array(test_k_all), np.array(labels_val_all))
        test_ami = metrics.adjusted_mutual_info_score(test_k_all, labels_val_all)

    results = {}
    results["NMI"] = test_nmi
    results["Purity"] = test_purity
    results["AMI"] = test_ami

    if np.abs(test_ami-0.) < 0.0001 and np.abs(test_nmi-0.125) < 0.0001:
        return None

    if data_set == "fMNIST":
        if convolution:
            f = open("results_fMNIST_conv.txt", "a+")
        else:
            f = open("results_fMNIST.txt", "a+")
    else:
        if convolution:
            f = open("results_MNIST_conv.txt", "a+")
        else:
            f = open("results_MNIST.txt", "a+")
    f.write(
            'Epochs= %d, som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, beta=%f, gamma=%f, '
            'theta=%f, alpha=%f, dropout=%f, decay_factor=%f, prior_var=%f, prior=%f, epochs_pretrain=%d'
            % (epochs, som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, beta,
               gamma, theta, alpha, dropout, decay_factor, prior_var, prior, epochs_pretrain))

    f.write(", RESULTS NMI: %f, AMI: %f, Purity: %f.  Name: %r \n"
            % (results["NMI"], results["AMI"], results["Purity"], ex_name))
    f.close()
    return results


@ex.automain
def main(latent_dim, som_dim, learning_rate, decay_factor, alpha, beta, gamma, theta, ex_name, more_runs, data_set,
         dropout, prior_var, convolution, prior, validation, epochs_pretrain, num_epochs, batch_size, random_seed,
         exp_output, exp_path):
    """Main method to build a model, train it and evaluate it.
    Returns:
        dict: Results of the evaluation (NMI, Purity).
    """
    random.seed(random_seed)
    nprand.seed(random_seed)
    tf.random.set_random_seed(random_seed)

    if exp_output:
        Path(os.path.join(exp_path, "exp_beta_{:.4f}_gamma_{:.4f}_bsize_{}_seed_{}_epochs_{}.LOCK".format(beta,gamma,batch_size,random_seed,num_epochs))).touch()

    start = time.time()
    if not os.path.exists('../models'):
        os.mkdir('../models')

    # Dimensions for MNIST-like data
    input_length = 28
    input_channels = 28
    lr_val = tf.placeholder_with_default(learning_rate, [])

    model = DPSOM(latent_dim=latent_dim, som_dim=som_dim, learning_rate=lr_val, alpha=alpha,
                    decay_factor=decay_factor, input_length=input_length, input_channels=input_channels, beta=beta,
                    theta=theta, gamma=gamma, convolution=convolution, dropout=dropout, prior_var=prior_var,
                    prior=prior)

    if data_set == "MNIST":
        mnist = tf.keras.datasets.mnist.load_data(path='mnist.npz')
        data_total = np.reshape(mnist[0][0], [-1, 28 * 28])
        maxx = np.reshape(np.amax(data_total, axis=-1), [-1, 1])
        data_total = np.reshape(data_total / maxx, [-1, 28, 28, 1])
        labels_total = mnist[0][1]
        data_test = np.reshape(mnist[1][0], [-1, 28 * 28])
        maxx = np.reshape(np.amax(data_test, axis=-1), [-1, 1])
        data_test = np.reshape(data_test / maxx, [-1, 28, 28, 1])
        labels_test = mnist[1][1]
        data_train, data_val, labels_train, labels_val = train_test_split(data_total, labels_total, test_size=0.15,
                                                                          random_state=42)

    else:
        ((data_total, labels_total), (data_test, labels_test)) = tf.keras.datasets.fashion_mnist.load_data()
        data_total = np.reshape(data_total, [-1, 28 * 28])
        maxx = np.reshape(np.amax(data_total, axis=-1), [-1, 1])
        data_total = np.reshape(data_total / maxx, [-1, 28, 28, 1])
        data_test = np.reshape(data_test, [-1, 28 * 28])
        maxx = np.reshape(np.amax(data_test, axis=-1), [-1, 1])
        data_test = np.reshape(data_test / maxx, [-1, 28, 28, 1])
        data_train, data_val, labels_train, labels_val = train_test_split(data_total, labels_total, test_size=0.15,
                                                                 random_state=42)
    data_generator = get_data_generator(data_train, data_val, labels_train, labels_val, data_test, labels_test)
    if not validation:
        data_val = data_test

    if more_runs:
        NMI = []
        PUR = []
        for i in range(10):
            results = train_model(model, data_train, data_val, data_generator, lr_val)
            NMI.append(results["NMI"])
            PUR.append(results["Purity"])
        NMI_mean = np.mean(NMI)
        NMI_sd = np.std(NMI) / np.sqrt(10)
        PUR_mean = np.mean(PUR)
        PUR_sd = np.std(PUR) / np.sqrt(10)
        print("\nRESULTS NMI: %f +- %f, PUR: %f +- %f.  Name: %r. \n" % (NMI_mean, NMI_sd, PUR_mean, PUR_sd, ex_name))
        if data_set == "MNIST":
            f = open("evaluation_MNIST.txt", "a+")
        else:
            f = open("evaluation_fMNIST.txt", "a+")
        f.write(
            "som_dim=[%d,%d], latent_dim= %d, batch_size= %d, learning_rate= %f, theta= %f, "
            "dropout=%f, prior=%f, gamma=%f, beta%f, epochs_pretrain=%d, epochs= %d"
            % (som_dim[0], som_dim[1], latent_dim, batch_size, learning_rate, theta, dropout, prior,
               gamma, beta, epochs_pretrain, num_epochs))

        f.write(", RESULTS NMI: %f + %f, PUR: %f + %f.  Name: %r \n"
                % (NMI_mean, NMI_sd, PUR_mean, PUR_sd, ex_name))
        f.close()
    else:
        results = train_model(model, data_train, data_val, data_generator, lr_val)
        print("\n NMI: {}, AMI: {}, PUR: {}.  Name: %r.\n".format(results["NMI"], results["AMI"], results["Purity"],
                                                                  ex_name))

        if exp_output:
            with open(os.path.join(exp_path, "exp_beta_{:.4f}_gamma_{:.4f}_bsize_{}_seed_{}_epochs_{}.tsv".format(beta,gamma,batch_size,random_seed,num_epochs)),'w') as out_fp:
                csv_fp=csv.writer(out_fp)
                csv_fp.writerow(["DATASET","NMI","AMI","Purity"])
                csv_fp.writerow([data_set,str(results["NMI"]), str(results["AMI"]), str(results["Purity"])])
            os.remove(os.path.join(exp_path, "exp_beta_{:.4f}_gamma_{:.4f}_bsize_{}_seed_{}_epochs_{}.LOCK".format(beta,gamma,batch_size,random_seed,num_epochs)))
        
        elapsed_time_fl = (time.time() - start)
        print("\n Time: {}".format(elapsed_time_fl))
    return results
