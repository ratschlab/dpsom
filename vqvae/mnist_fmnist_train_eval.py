
import os
from glob import glob
import pickle
import argparse
import ipdb
import csv
import random

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

    x = tf.placeholder(tf.float32, shape=[None, 784])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    y = tf.placeholder(tf.int32, shape=[None])
    train = tf.placeholder(tf.bool, name="train")
    batch_size = tf.shape(x)[0]

    with tf.variable_scope("encoder"):
        h_conv1 = tf.nn.relu(conv2d(x_image, [4,4,1,configs["conv_size"]], "conv1"))
        h_pool1 = max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(conv2d(h_pool1, [4,4,configs["conv_size"],configs["conv_size"]], "conv2"))
        h_pool2 = max_pool_2x2(h_conv2)
        flat_size = 7*7*configs["conv_size"]
        h_flat = tf.reshape(h_pool2, [batch_size, flat_size])
    #     h_flat_norm = tf.layers.batch_normalization(h_flat, training=train, renorm=True)
        z_e = tf.keras.layers.Dense(latent_dim)(h_flat)


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
            h_flat_dec = tf.keras.layers.Dense(flat_size)(z_tensor)
            h_reshaped = tf.reshape(h_flat_dec, tf.shape(h_pool2))
            h_unpool1 = tf.keras.layers.UpSampling2D((2,2))(h_reshaped)
            h_deconv1 = tf.nn.relu(conv2d(h_unpool1, [4,4,configs["conv_size"],configs["conv_size"]], "deconv1"))
            h_unpool2 = tf.keras.layers.UpSampling2D((2,2))(h_deconv1)
            h_deconv2 = tf.nn.sigmoid(conv2d(h_unpool2, [4,4,configs["conv_size"],1], "deconv2"))
            x_hat = h_deconv2
            return x_hat

    x_hat = decoder(z_q)

    beta = 0.25
    loss_rec_mse = tf.losses.mean_squared_error(x_image, x_hat)
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

    for data_set in configs["DATASETS"]:

        if data_set=="mnist":
            ds_train,ds_test=tf.keras.datasets.mnist.load_data()
        elif data_set=="fashion":
            ds_train,ds_test=tf.keras.datasets.fashion_mnist.load_data()

        data_train=ds_train[0]
        data_train=np.reshape(data_train, (data_train.shape[0], data_train.shape[1]*data_train.shape[2]))
        data_test=ds_test[0]
        data_test=np.reshape(data_test,(data_test.shape[0],data_test.shape[1]*data_test.shape[2]))
        labels_test =ds_test[1]
        labels_train =ds_train[1]
        aggregated_mses = []
        aggregated_NMIs = []
        aggregated_purities = []

        for _ in range(NUM_TESTS):
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                indices_unsup = np.arange(data_train.shape[0])
                with tqdm(total=EPOCHS*(data_train.shape[0]//BATCH_SIZE)) as pbar:
                    for epoch in range(EPOCHS):
                        np.random.shuffle(indices_unsup)
                        test_mse = sess.run(loss_rec_mse, feed_dict={x: data_test[:100], train: False})
                        for i in range(indices_unsup.shape[0]//BATCH_SIZE):
                            batch_data = data_train[indices_unsup[BATCH_SIZE*i:BATCH_SIZE*(i+1)]]
                            if i%100 == 0:
                                train_mse, train_commit, train_loss = sess.run([loss_rec_mse, loss_commit, loss],
                                                                               feed_dict={x: batch_data, train: False})
                            train_step.run(feed_dict={x: batch_data, train: True})
                            pbar.set_postfix(epoch=epoch, train_mse=train_mse, train_commit=train_commit,
                                                 test_mse=test_mse, refresh=False)
                            pbar.update(1)

                test_k_all = []
                test_x_hat_all = []
                for i in trange(data_test.shape[0]//100):
                    batch_data = data_test[100*i:100*(i+1)]
                    test_k_all.extend(sess.run(k, feed_dict={x: batch_data, train: False}))
                    test_x_hat_all.extend(sess.run(x_hat, feed_dict={x: batch_data, train: False}))
                test_x_hat_all = np.array(test_x_hat_all)
                test_k_all=np.array(test_k_all)

            aggregated_mses.append(mean_squared_error(data_test, np.reshape(test_x_hat_all, [10000, 784])))
            aggregated_NMIs.append(normalized_mutual_info_score(test_k_all, labels_test[:len(test_k_all)]))
            aggregated_purities.append(cluster_purity(test_k_all, labels_test[:len(test_k_all)]))

        print("Results for {}".format(data_set)) 
        print("Test MSE: {} +- {}\nTest NMI: {} +- {}\nTest purity: {} +- {}".format(np.mean(aggregated_mses),
                    np.std(aggregated_mses)/np.sqrt(NUM_TESTS), np.mean(aggregated_NMIs), np.std(aggregated_NMIs)/
                    np.sqrt(NUM_TESTS), np.mean(aggregated_purities), np.std(aggregated_purities)/np.sqrt(NUM_TESTS)))

        if not configs["debug_mode"]:
            with open("../results/vqvae_{}_{}_somdim_{}.tsv".format(data_set,configs["random_state"], configs["som_dim"]),'w') as fp:
                csv_fp=csv.writer(fp,delimiter='\t')
                csv_fp.writerow(["model","mse","nmi","purity"])
                csv_fp.writerow(["vqvae",str(aggregated_mses[0]), str(aggregated_NMIs[0]), str(aggregated_purities[0])])

def parse_cmd_args():
    parser=argparse.ArgumentParser()

    parser.add_argument("--latent_dim", type=int, default=16, help="Latent dimension of VQ-VAE") 
    parser.add_argument("--conv_size", type=int, default=32, help="Size of conv layers")
    parser.add_argument("--som_dim", type=int, default=16, help="Grid size on one side") 
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=20, help="Number of epochs to train for") 
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--debug_mode", action="store_true", default=False, help="No output to FS")

    configs=vars(parser.parse_args())

    configs["DATASETS"]=["mnist","fashion"]

    return configs


if __name__=="__main__":
    configs=parse_cmd_args()
    execute(configs)
