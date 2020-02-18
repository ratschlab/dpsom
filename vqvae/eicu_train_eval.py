''' VQ-VAE training/evaluation on the eICU data-set'''

import os
from glob import glob
import pickle
import argparse
import ipdb
import csv
import random
import sys
import timeit

import numpy as np
import numpy.random as nprand
import h5py
from tqdm import tqdm, trange

from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display

try:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from matplotlib.colors import ListedColormap
import seaborn as sns
sns.set(style="white", context="paper")

from sklearn.metrics import mean_squared_error, normalized_mutual_info_score,accuracy_score
from sklearn.model_selection import train_test_split

def get_data(test=True, train_ratio=1.0):
    ''' Get the precomputed data from the file-system'''
    hf = h5py.File(configs["eicu_data"], 'r')
    data_total = np.array(hf.get('x'))
    endpoints_total = np.array(hf.get('y'))
    hf.close()
    data_train, data_val, y_train, endpoints_total_val = train_test_split(data_total[:int(len(data_total) * 0.85)],
                                                                          endpoints_total[:int(len(data_total) * 0.85)],
                                                                          test_size=0.20,
                                                                          random_state=42)

    if train_ratio<1.0:
        data_train=data_train[:int(len(data_train)*train_ratio)]
        y_train=y_train[:int(len(y_train)*train_ratio)]

    if test:
        data_val = data_total[int(len(data_total) * 0.85):]
        endpoints_total_val = endpoints_total[int(len(data_total) * 0.85):]

    data_train=np.reshape(data_train,(data_train.shape[0]*data_train.shape[1],-1))
    data_val=np.reshape(data_val,(data_val.shape[0]*data_val.shape[1],-1))

    y_train=np.reshape(y_train,(y_train.shape[0]*y_train.shape[1],-1))
    endpoints_total_val=np.reshape(endpoints_total_val,(endpoints_total_val.shape[0]*endpoints_total_val.shape[1],-1))
    return data_train, data_val, y_train, endpoints_total_val

def cluster_purity(y_pred,y_true):
    """
    Calculate clustering purity
    # Arguments
        y_true: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        purity, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_pred.copy()
    for i in range(y_pred.size):
        y_pred_voted[i] = label_mapping[y_pred[i]]
    return accuracy_score(y_pred_voted, y_true)


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial,name=name)

def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, shape, name, strides=[1,1,1,1]):
    weight = weight_variable(shape, "{}_W".format(name))
    bias = bias_variable([shape[-1]], "{}_b".format(name))
    return tf.nn.conv2d(x, weight, strides=strides, padding='SAME', name=name) + bias

def conv2d_transposed(x, shape, outshape, name, strides=[1,1,1,1]):
    weight = weight_variable(shape, "{}_W".format(name))
    bias = bias_variable([shape[-2]], "{}_b".format(name))
    return tf.nn.conv2d_transpose(x, weight, output_shape=outshape, strides=strides, padding='SAME', name=name) + bias

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def execute(configs):
    tf.reset_default_graph()
    random.seed(configs["random_state"])
    nprand.seed(configs["random_state"])

    DECAY_FACTOR = 0.80
    decay_steps = 1000
    latent_dim = configs["latent_dim"]
    som_dim = [configs["som_dim"], configs["som_dim"]]
    num_classes = 10
    global_step = tf.Variable(0, trainable=False, name="global_step")

    embeddings = tf.get_variable("embeddings", som_dim+[latent_dim],
                                 initializer=tf.truncated_normal_initializer(stddev=0.05))

    x_patient = tf.placeholder(tf.float32, shape=[None, 98])
    y = tf.placeholder(tf.int32, shape=[None])
    train = tf.placeholder(tf.bool, name="train")
    batch_size = tf.shape(x_patient)[0]

    with tf.variable_scope("encoder"):
        dense_1=tf.keras.layers.Dense(configs["conv_size"])(x_patient)
        dense_2=tf.keras.layers.Dense(configs["conv_size"])(dense_1)
        z_e = tf.keras.layers.Dense(latent_dim)(dense_2)

    z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(z_e, 1), 1), tf.expand_dims(embeddings, 0))
    z_dist_red = tf.reduce_sum(z_dist, axis=-1)
    z_dist_flat = tf.reshape(z_dist_red, [batch_size, -1])
    k = tf.argmin(z_dist_flat, axis=-1)
    k_1 = k // som_dim[1]
    k_2 = k % som_dim[1]
    k_stacked = tf.stack([k_1, k_2], axis=1)
    z_q = tf.gather_nd(embeddings, k_stacked)

    def decoder(z_tensor):
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            dec_dense_1 = tf.keras.layers.Dense(configs["conv_size"])(z_tensor)
            dec_dense_2 = tf.keras.layers.Dense(configs["conv_size"])(dec_dense_1)
            flat_dec=tf.keras.layers.Dense(98)(dec_dense_2)
            x_hat = flat_dec
            return x_hat

    x_hat = decoder(z_q)

    beta = 0.25
    loss_rec_mse = tf.losses.mean_squared_error(x_patient, x_hat)
    loss_vq = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(z_e), z_q))
    loss_commit = tf.reduce_mean(tf.squared_difference(z_e, tf.stop_gradient(z_q)))
    loss = loss_rec_mse + loss_vq + beta*loss_commit

    learning_rate = tf.placeholder_with_default(0.001, [])
    lr_decay = tf.train.exponential_decay(learning_rate, global_step, decay_steps, DECAY_FACTOR, staircase=True)

    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "decoder")
    decoder_grads = list(zip(tf.gradients(loss, decoder_vars), decoder_vars))
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "encoder")
    grad_z = tf.gradients(loss_rec_mse, z_q)

    encoder_grads = [(tf.gradients(z_e,var,grad_z)[0]+beta*tf.gradients(loss_commit,var)[0],var) for var in encoder_vars]
    embed_grads = list(zip(tf.gradients(loss_vq, embeddings),[embeddings]))

    optimizer = tf.train.AdamOptimizer(lr_decay)
    train_step = optimizer.apply_gradients(decoder_grads+encoder_grads+embed_grads)

    BATCH_SIZE = configs["batch_size"]
    EPOCHS = configs["n_epochs"]
    NUM_TESTS = 1 

    if configs["benchmark"]:
        times_per_epoch=[]

    for data_set in configs["DATASETS"]:

        if not configs["debug_mode"]:
            with open("../results/vqvae_{}_{}.tsv".format(data_set,configs["random_state"]),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["model","task","nmi"])

        if data_set=="eicu":
            data_train, data_test, labels_train, labels_test = get_data(test=True, train_ratio=configs["train_ratio"])

        for _ in range(NUM_TESTS):

            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                indices_unsup = np.arange(data_train.shape[0])
                with tqdm(total=EPOCHS*(data_train.shape[0]//BATCH_SIZE)) as pbar:
                    for epoch in range(EPOCHS):
                        if configs["benchmark"]:
                            t_begin=timeit.default_timer()
                        np.random.shuffle(indices_unsup)
                        test_mse = sess.run(loss_rec_mse, feed_dict={x_patient: data_test[:100], train: False})
                        for i in range(indices_unsup.shape[0]//BATCH_SIZE):
                            batch_data = data_train[indices_unsup[BATCH_SIZE*i:BATCH_SIZE*(i+1)]]
                            if i%100 == 0:
                                train_mse, train_commit, train_loss = sess.run([loss_rec_mse, loss_commit, loss],
                                                                               feed_dict={x_patient: batch_data, train: False})
                            train_step.run(feed_dict={x_patient: batch_data, train: True})
                            pbar.set_postfix(epoch=epoch, train_mse=train_mse, train_commit=train_commit,
                                                 test_mse=test_mse, refresh=False)
                            pbar.update(1)

                        if configs["benchmark"]:
                            t_end=timeit.default_timer()
                            times_per_epoch.append(t_end-t_begin)

                if configs["benchmark"]:
                    print("Times per epoch: {:.3f}".format(np.mean(times_per_epoch)))
                    sys.exit(0)

                test_k_all = []
                test_x_hat_all = []
                for i in trange(data_test.shape[0]//100):
                    batch_data = data_test[100*i:100*(i+1)]
                    test_k_all.extend(sess.run(k, feed_dict={x_patient: batch_data, train: False}))
                    test_x_hat_all.extend(sess.run(x_hat, feed_dict={x_patient: batch_data, train: False}))
                test_x_hat_all = np.array(test_x_hat_all)
                test_k_all=np.array(test_k_all)

            data_test=data_test[:test_x_hat_all.shape[0]] 

            for task_desc,task_idx in [("apache_0",0), ("apache_6",1), ("apache_12",2), ("apache_24",3)]:
                labels_test_task=labels_test[:,task_idx]
                aggregated_mses = []
                aggregated_NMIs = []
                aggregated_purities = []
                aggregated_mses.append(mean_squared_error(data_test, test_x_hat_all))
                aggregated_NMIs.append(normalized_mutual_info_score(test_k_all, labels_test_task[:len(test_k_all)]))
                aggregated_purities.append(cluster_purity(test_k_all, labels_test_task[:len(test_k_all)]))

                print("Results for {} on task: {}".format(data_set,task_desc))
                print("Test MSE: {} +- {}\nTest NMI: {} +- {}\nTest purity: {} +- {}".format(np.mean(aggregated_mses),
                            np.std(aggregated_mses)/np.sqrt(NUM_TESTS), np.mean(aggregated_NMIs), np.std(aggregated_NMIs)/
                            np.sqrt(NUM_TESTS), np.mean(aggregated_purities), np.std(aggregated_purities)/np.sqrt(NUM_TESTS)))

                if not configs["debug_mode"]:
                    with open("../results/vqvae_{}_{}.tsv".format(data_set,configs["random_state"]),'a') as fp:
                        csv_fp=csv.writer(fp,delimiter='\t')
                        csv_fp.writerow(["vqvae",task_desc,str(aggregated_NMIs[0])])

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of VQ-VAE")
    parser.add_argument("--conv_size", type=int, default=32, help="Size of conv layers")
    parser.add_argument("--som_dim", type=int, default=16, help="Grid size on one side") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs to train for") 
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="No output to FS")
    parser.add_argument("--benchmark", default=True, action="store_true", help="Benchmark mode?")
    parser.add_argument("--train_ratio", default=0.5, type=float, help="Subset of training data to use")

    # Input paths
    parser.add_argument("--eicu_data", default="../data/eICU_data.csv")

    configs=vars(parser.parse_args())

    configs["DATASETS"]=["eicu"]

    return configs


if __name__=="__main__":
    configs=parse_cmd_args()
    execute(configs)
