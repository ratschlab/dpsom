"""
VarIDEC model
"""

import functools
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization
from sklearn.cluster import KMeans


def lazy_scope(function):
    """Creates a decorator for methods that makes their return values load lazily.
    
    A method with this decorator will only compute the return value once when called
    for the first time. Afterwards, the value will be cached as an object attribute.
    Inspired by: https://danijar.com/structuring-your-tensorflow-models
    
    Args:
        function (func): Function to be decorated.
        
    Returns:
        decorator: Decorator for the function.
    """
    attribute = "_cache_" + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class VarIDEC:
    """Class for the SOM-VAE model as described in https://arxiv.org/abs/1806.02199"""

    def __init__(self, latent_dim=64, num_clusters=10, learning_rate=1e-4, decay_factor=0.99, decay_steps=2000,
                 input_length=28, input_channels=28, alpha=1., gamma=1., theta=1., dropout=0.4, prior_var=1, prior=0.5):
        """Initialization method for the SOM-VAE model object.
        Args:
            inputs (tf.Tensor): The input tensor for the model.
            latent_dim (int): The dimensionality of the latent embeddings (default: 64).
            som_dim (list): The dimensionality of the self-organizing map (default: [8,8]).
            learning_rate (float): The learning rate for the optimization (default: 1e-4).
            decay_factor (float): The factor for the learning rate decay (default: 0.95).
            decay_steps (int): The number of optimization steps before every learning rate
                decay (default: 1000).
            input_length (int): The length of the input data points (default: 28).
            input_channels (int): The number of channels of the input data points (default: 28).
            alpha (float): The weight for the commitment loss (default: 1.).
            beta (float): The weight for the SOM loss (default: 1.).
            gamma (float): The weight for the transition probability loss (default: 1.).
            tau (float): The weight for the smoothness loss (default: 1.).
            mnist (bool): Flag that tells the model if we are training in MNIST-like data (default: True).
        """
        self.latent_dim = latent_dim
        self.num_clusters = num_clusters
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_length = input_length
        self.input_channels = input_channels
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.dropout = dropout
        self.prior_var = prior_var
        self.prior = prior
        self.is_training
        self.inputs
        self.input_dim
        self.batch_size
        self.embeddings
        self.global_step
        self.z_e
        self.sample_z_e
        self.z_dist_flat
        self.k
        self.reconstruction_e
        self.loss_reconstruction_ze
        self.p
        self.q
        self.loss_commit
        self.loss
        self.optimize

    @lazy_scope
    def inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
        return x

    @lazy_scope
    def is_training(self):
        is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        return is_training

    @lazy_scope
    def input_dim(self):
        x = 28 * 28
        return x

    @lazy_scope
    def batch_size(self):
        """Reads the batch size from the input tensor."""
        batch_size = tf.shape(self.inputs)[0]
        return batch_size

    @lazy_scope
    def embeddings(self):
        """Creates variable for the clusters centroid."""
        embeddings = tf.get_variable("embeddings",[self.num_clusters,self.latent_dim],
                             initializer=tf.truncated_normal_initializer(stddev=0.05))
        tf.summary.tensor_summary("embeddings", embeddings)
        return embeddings

    @lazy_scope
    def global_step(self):
        """Creates global_step variable for the optimization."""
        global_step = tf.Variable(0, trainable=False, name="global_step")
        return global_step

    @lazy_scope
    def z_e(self):
        """Computes the distribution of probability of the latent embeddings."""
        with tf.variable_scope("encoder"):
            inputs = tf.reshape(self.inputs, (-1, self.input_dim))
            dense1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(inputs)
            dense1 = tf.keras.layers.Dropout(rate=self.dropout)(dense1)
            dense1 = BatchNormalization()(dense1)
            dense2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense1)
            dense2 = tf.keras.layers.Dropout(rate=self.dropout)(dense2)
            dense2 = BatchNormalization()(dense2)
            dense3 = tf.keras.layers.Dense(2000, activation=tf.nn.leaky_relu)(dense2)
            dense3 = tf.keras.layers.Dropout(rate=self.dropout)(dense3)
            dense3 = BatchNormalization()(dense3)
            z_e = Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim), activation=None)(
                    dense3)
            z_e = tfp.layers.MultivariateNormalTriL(self.latent_dim)(z_e)
        return z_e

    @lazy_scope
    def sample_z_e(self):
        sample_z_e = tf.identity(self.z_e, name="z_e")
        return sample_z_e

    @lazy_scope
    def z_dist_flat(self):
        """Computes the distances between the embeddings and the centroids."""
        z_dist = tf.squared_difference(tf.expand_dims(self.z_e, 1), self.embeddings)
        z_dist_red = tf.reduce_sum(z_dist, axis=2)
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.num_clusters])
        return z_dist_flat

    @lazy_scope
    def k(self):
        """Picks the index of the closest centroid for every embedding."""
        k = tf.argmin(self.z_dist_flat, axis=-1, name="k")
        tf.summary.histogram("clusters", k)
        return k

    @lazy_scope
    def reconstruction_e(self):
        """Reconstructs the input from the encodings by learning a Bernoulli distribution."""
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            flat_size = 2000
            z_p = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name="z_e")
            z_e = tf.cond(self.is_training, lambda: self.z_e, lambda: z_p)
            dense4 = tf.keras.layers.Dense(flat_size, activation=tf.nn.leaky_relu)(z_e)
            dense4 = BatchNormalization()(dense4)
            dense3 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense4)
            dense3 = BatchNormalization()(dense3)
            dense2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense3)
            dense2 = BatchNormalization()(dense2)
            dense1 = tf.keras.layers.Dense(self.input_dim, activation=tf.nn.leaky_relu)(dense2)
            x_hat = tfp.layers.IndependentBernoulli([28, 28, 1], tfp.distributions.Bernoulli.logits)(dense1)
            x_hat_sampled = tf.identity(x_hat, name="x_hat")
        return x_hat

    @lazy_scope
    def loss_reconstruction_ze(self):
        """Computes the ELBO."""
        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim),
                                                         scale_diag=tf.ones(self.latent_dim)*self.prior_var)
        kl_loss = tf.reduce_mean(self.z_e.kl_divergence(prior))
        tf.summary.scalar("loss_reconstruction_kl", kl_loss)
        log_lik_loss = -tf.reduce_mean(self.reconstruction_e.log_prob(self.inputs))
        tf.summary.scalar("loss_reconstruction_log_lik_loss", log_lik_loss)
        loss_rec_mse_ze = self.prior*kl_loss + log_lik_loss
        loss_rec_mse = loss_rec_mse_ze
        return loss_rec_mse

    @lazy_scope
    def q(self):
        """Computes the target distribution given the soft assignment between embeddings and centroids."""
        with tf.name_scope('distribution'):
            q = 1.0 / (1.0 + self.z_dist_flat / self.alpha) ** ((self.alpha + 1.0) / 2.0)
            q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
            q = tf.identity(q, name="q")
        return q

    @lazy_scope
    def p(self):
        """Placeholder for the target distribution."""
        p = tf.placeholder(tf.float32, shape=(None, self.num_clusters))
        return p

    @lazy_scope
    def loss_commit(self):
        """Computes the commitment loss."""
        loss_commit = tf.reduce_mean(tf.reduce_sum(self.p * tf.log(self.p / (self.q)), axis=1))
        tf.summary.scalar("loss_commit", loss_commit)
        return loss_commit

    def target_distribution(self, q):
        """Computes the target distribution given the soft assignment between embeddings and centroids."""
        p = q ** 2 / q.sum(axis=0)
        p = p / p.sum(axis=1, keepdims=True)
        return p

    def get_assign_cluster_centers_op(self, features):
        # init embeddings
        print('Start training KMeans')
        k = KMeans(n_clusters=self.num_clusters, n_init=20)
        kmeans = k.fit(features.reshape(len(features), self.latent_dim))
        print('Finish training KMeans')
        return tf.assign(self.embeddings, kmeans.cluster_centers_.reshape(self.num_clusters, self.latent_dim))

    @lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss."""
        loss = self.theta*self.loss_reconstruction_ze + self.gamma*self.loss_commit
        tf.summary.scalar("loss", loss)
        return loss

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        train_step_ae = optimizer.minimize(self.loss_reconstruction_ze, global_step=self.global_step)
        return train_step, train_step_ae
