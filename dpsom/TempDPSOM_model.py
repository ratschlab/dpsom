"""
Model architecture of TempDPSOM model
"""

import functools
import numpy as np

try:
    import tensorflow.compat.v1 as tf 
    tf.disable_v2_behavior()
except:
    import tensorflow as tf

import tensorflow_probability as tfp
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization


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


class TDPSOM:
    """Class for the T-DPSOM model"""

    def __init__(self, input_size, latent_dim=10, som_dim=[8, 8], learning_rate=1e-4, decay_factor=0.99,
                 decay_steps=2000, input_channels=98, alpha=10., beta=100., gamma=100., kappa=0.,
                 theta=1., eta=1., dropout=0.5, prior=0.001, lstm_dim=100):

        """Initialization method for the T-DPSOM model object.
        Args:
            input_size (int): Length of the input vector.
            latent_dim (int): The dimensionality of the latent embeddings (default: 100).
            som_dim (list): The dimensionality of the self-organizing map (default: [8,8]).
            learning_rate (float): The learning rate for the optimization (default: 1e-4).
            decay_factor (float): The factor for the learning rate decay (default: 0.99).
            decay_steps (int): The number of optimization steps before every learning rate
                decay (default: 1000).
            input_channels (int): The number of channels of the input data points (default: 98).
            alpha (float): The weight for the commitment loss (default: 10.).
            beta (float): Weight for the SOM loss (default: 100).
            gamma (float): Weight for the KL term of the PSOM clustering loss (default: 100).
            kappa (float): Weight for the smoothness loss (default: 10).
            theta (float): Weight for the VAE loss (default: 1).
            eta (float): Weight for the prediction loss (default: 1).
            dropout (float): Dropout factor for the feed-forward layers of the VAE (default: 0.5).
            prior (float): Weight of the regularization term of the ELBO (default: 0.5).
        """

        self.input_size = input_size
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_channels = input_channels
        self.alpha = alpha
        self.beta = beta
        self.eta = eta
        self.gamma = gamma
        self.theta = theta
        self.kappa = kappa
        self.dropout = dropout
        self.prior = prior
        self.lstm_dim = lstm_dim
        self.prior
        self.is_training
        self.inputs
        self.x
        self.batch_size
        self.step_size
        self.embeddings
        self.global_step
        self.z_e
        self.z_e_sample
        self.k
        self.prediction
        self.z_e_old
        self.z_dist_flat
        self.z_q
        self.z_q_neighbors
        self.reconstruction_e
        self.loss_reconstruction_ze
        self.q
        self.p
        self.loss_commit
        self.loss_som
        self.loss_prediction
        self.loss_smoothness
        self.loss
        self.loss_a
        self.optimize

    @lazy_scope
    def is_training(self):
        is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        return is_training

    @lazy_scope
    def inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, None, 98], name="x")
        return x

    @lazy_scope
    def x(self):
        x = tf.reshape(self.inputs, [-1, 98])
        return x

    @lazy_scope
    def batch_size(self):
        """Reads the batch size from the input tensor."""
        batch_size = tf.shape(self.inputs)[0]
        return batch_size

    @lazy_scope
    def step_size(self):
        """Reads the step size from the input tensor."""
        step_size = tf.shape(self.inputs)[1]
        return step_size

    @lazy_scope
    def embeddings(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.get_variable("embeddings", self.som_dim + [self.latent_dim],
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
            h_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(self.x)
            h_1 = tf.keras.layers.Dropout(rate=self.dropout)(h_1)
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(h_1)
            h_1 = tf.keras.layers.Dropout(rate=self.dropout)(h_1)
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_2 = tf.keras.layers.Dense(2000, activation=tf.nn.leaky_relu)(h_1)
            h_2 = tf.keras.layers.Dropout(rate=self.dropout)(h_2)
            h_2 = tf.keras.layers.BatchNormalization()(h_2)
            z_e_mu = tf.keras.layers.Dense(self.latent_dim, activation=None)(h_2)
            z_e_sigma = tf.keras.layers.Dense(self.latent_dim, activation=None)(h_2)
            z_e = tfp.distributions.MultivariateNormalDiag(loc=z_e_mu, scale_diag=tfp.bijectors.Softplus()(z_e_sigma))
            #z_e = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim),activation=None)(h_2)
            #z_e = tfp.layers.MultivariateNormalTriL(self.latent_dim)(z_e)
        return z_e

    @lazy_scope
    def z_e_sample(self):
        """Sample from the distribution of probability of the latent embeddings."""
        #z_e = tf.identity(self.z_e, name="z_e")
        z_e = self.z_e.sample()
        z_e = tf.identity(z_e, name="z_e")
        tf.summary.histogram("count_nonzeros_z_e", tf.count_nonzero(z_e, -1))
        return z_e

    @lazy_scope
    def z_e_old(self):
        """Aggregates the encodings of the respective previous time steps."""
        z_e_old = tf.concat([self.z_e_sample[0:1], self.z_e_sample[:-1]], axis=0)
        return z_e_old

    @lazy_scope
    def z_dist_flat(self):
        """Computes the distances between the centroids and the embeddings."""
        z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(self.z_e_sample, 1), 1),
                                       tf.expand_dims(self.embeddings, 0))
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)  # 1,32,8,8
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[0]])  # 1,32,64
        return z_dist_flat

    @lazy_scope
    def z_dist_flat_ng(self):
        """Computes the distances between the centroids and the embeddings stopping the gradient of the latent
        embeddings."""
        z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(tf.stop_gradient(self.z_e_sample), 1), 1),
                                       tf.expand_dims(self.embeddings, 0))
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)  # 1,32,8,8
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[1]])  # 1,32,64
        return z_dist_flat

    @lazy_scope
    def k(self):
        """Picks the index of the closest centroid for every embedding."""
        k = tf.argmin(self.z_dist_flat, axis=-1, name="k")
        tf.summary.histogram("clusters", k)
        return k

    @lazy_scope
    def prediction(self):
        """Predict the distribution of probability of the next embedding."""
        with tf.variable_scope("next_state"):
            z_e_p = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name="input_lstm")
            z_e = tf.cond(self.is_training, lambda: self.z_e_sample, lambda: z_e_p)

            rnn_input = tf.stop_gradient(tf.reshape(z_e, [self.batch_size, self.step_size, self.latent_dim]))
            init_state_p = tf.placeholder(tf.float32, shape=[2, None, self.lstm_dim], name="init_state")

            cell = tf.keras.layers.LSTM(self.lstm_dim, return_sequences=True, return_state=True)
            init_state = cell.get_initial_state(rnn_input)
            state = tf.cond(self.is_training, lambda: init_state, lambda: [init_state_p[0], init_state_p[1]])
            lstm_output, state_h, state_c = cell(rnn_input, initial_state=state)
            state = tf.identity([state_h, state_c], name="next_state")
            lstm_output = tf.reshape(lstm_output, [self.batch_size*self.step_size, self.lstm_dim])

            h_1 = tf.keras.layers.Dense(self.lstm_dim, activation=tf.nn.leaky_relu)(lstm_output)
            next_z_e = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self.latent_dim),
                                            activation=None)(h_1)
            next_z_e = tfp.layers.IndependentNormal(self.latent_dim)(next_z_e)
            #next_z_e = tf.keras.layers.Dense(self.latent_dim, activation=None)(h_1)

        next_z_e_sample = tf.reshape(tf.identity(next_z_e), [-1, self.step_size, self.latent_dim], name="next_z_e")
        return next_z_e

    @lazy_scope
    def z_q(self):
        """Aggregates the respective closest embedding for every centroid."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)
        z_q = tf.gather_nd(self.embeddings, k_stacked, name="z_q")  # dim64
        return z_q

    @lazy_scope
    def z_q_neighbors(self):
        """Aggregates the respective neighbors in the SOM grid for every z_q."""
        k_1 = self.k // self.som_dim[1]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int64))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int64))

        k1_up = tf.where(k1_not_top, tf.add(k_1, 1), tf.zeros(tf.shape(k_1), dtype=tf.dtypes.int64))
        k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1),
                           tf.ones(tf.shape(k_1), dtype=tf.dtypes.int64) * (self.som_dim[0] - 1))
        k2_right = tf.where(k2_not_right, tf.add(k_2, 1), tf.zeros(tf.shape(k_2), dtype=tf.dtypes.int64))
        k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1),
                           tf.ones(tf.shape(k_2), dtype=tf.dtypes.int64) * (self.som_dim[0] - 1))

        z_q_up = tf.gather_nd(self.embeddings, tf.stack([k1_up, k_2], axis=1))
        z_q_down = tf.gather_nd(self.embeddings, tf.stack([k1_down, k_2], axis=1))
        z_q_right = tf.gather_nd(self.embeddings, tf.stack([k_1, k2_right], axis=1))
        z_q_left = tf.gather_nd(self.embeddings, tf.stack([k_1, k2_left], axis=1))

        z_q_neighbors = tf.stack([self.z_q, z_q_up, z_q_down, z_q_right, z_q_left], axis=1)
        return z_q_neighbors

    @lazy_scope
    def reconstruction_e(self):
        """Reconstructs the input from the encodings by learning a Gaussian distribution."""
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            z_p = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name="z_e")
            z_e = tf.cond(self.is_training, lambda: self.z_e_sample, lambda: z_p)

            h_1 = tf.keras.layers.Dense(2000, activation=tf.nn.leaky_relu)(z_e)
            h_1 = tf.keras.layers.BatchNormalization()(h_1)
            h_2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(h_1)
            h_2 = tf.keras.layers.BatchNormalization()(h_2)
            h_3 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(h_2)
            h_3 = tf.keras.layers.BatchNormalization()(h_3)

            #x_hat = tf.keras.layers.Dense(self.input_channels)(h_3)
            x_hat = tf.keras.layers.Dense(tfp.layers.IndependentNormal.params_size(self.input_channels),
                                                activation=None)(h_3)
            x_hat = tfp.layers.IndependentNormal(self.input_channels)(x_hat)
            #x_hat = tf.keras.layers.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.input_channels),
            #                            activation=None)(h_3)
            #x_hat = tfp.layers.MultivariateNormalTriL(self.input_channels)(x_hat)
        x_hat_sampled = tf.identity(x_hat, name="x_hat")
        return x_hat

    @lazy_scope
    def loss_reconstruction_ze(self):
        """Computes the ELBO."""
        prior = tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.latent_dim), scale_diag=tf.ones(self.latent_dim))
        kl_loss = tf.reduce_mean(self.z_e.kl_divergence(prior))
        log_lik_loss = -tf.reduce_mean(self.reconstruction_e.log_prob(self.x))
        #log_lik_loss = tf.keras.losses.MeanSquaredError()(self.reconstruction_e, self.x)
        loss_rec_mse_ze = self.prior * kl_loss + log_lik_loss
        tf.summary.scalar("log_lik_loss", log_lik_loss)
        tf.summary.scalar("kl_loss", kl_loss)
        tf.summary.scalar("loss_reconstruction_ze", loss_rec_mse_ze)
        return loss_rec_mse_ze

    @lazy_scope
    def q(self):
        """Computes the soft assignments between the embeddings and the centroids."""
        with tf.name_scope('distribution'):
            q = tf.keras.backend.epsilon() + 1.0 / (1.0 + self.z_dist_flat / self.alpha) ** ((self.alpha + 1.0) / 2.0)
            q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
            q = tf.identity(q, name="q")
        return q

    @lazy_scope
    def q_ng(self):
        """Computes the soft assignments between the embeddings and the centroids stopping the gradient of the latent
        embeddings."""
        with tf.name_scope('distribution'):
            q = tf.keras.backend.epsilon() + 1.0 / (1.0 + self.z_dist_flat_ng / self.alpha) ** ((self.alpha + 1.0) / 2.0)
            q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    @lazy_scope
    def p(self):
        """Placeholder for the target distribution."""
        p = tf.placeholder(tf.float32, shape=(None, self.som_dim[0] * self.som_dim[1]))
        return p

    @lazy_scope
    def loss_commit(self):
        """Computes the KL term of the clustering loss."""
        loss_commit = tf.reduce_mean(tf.reduce_sum(self.p * tf.log((self.p) / (self.q)), axis=1))
        return loss_commit

    def target_distribution(self, q):
        """Computes the target distribution given the soft assignment between embeddings and centroids."""
        p = q ** 2 / (q.sum(axis=0))
        p = p / p.sum(axis=1, keepdims=True)
        return p

    @lazy_scope
    def loss_som(self):
        """Computes the SOM loss."""
        k = tf.range(self.som_dim[0] * self.som_dim[1])
        k_1 = k // self.som_dim[0]
        k_2 = k % self.som_dim[1]

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0] - 1, dtype=tf.int32))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int32))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1] - 1, dtype=tf.int32))
        k2_not_left = tf.greater(k_2, tf.constant(0, dtype=tf.int32))

        k1_up = tf.where(k1_not_top, tf.add(k_1, 1), tf.zeros(tf.shape(k_1), dtype=tf.dtypes.int32))
        k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1), tf.ones(tf.shape(k_1), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1))
        k2_right = tf.where(k2_not_right, tf.add(k_2, 1), tf.zeros(tf.shape(k_2), dtype=tf.dtypes.int32))
        k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1), tf.ones(tf.shape(k_2), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1))

        k_up = k1_up * self.som_dim[0] + k_2
        k_down = k1_down * self.som_dim[0] + k_2
        k_right = k_1 * self.som_dim[0] + k2_right
        k_left = k_1 * self.som_dim[0] + k2_left

        q_t = tf.transpose(self.q_ng)
        q_up = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_up, [self.som_dim[0] * self.som_dim[1], 1])))
        q_down = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_down, [self.som_dim[0] * self.som_dim[1], 1])))
        q_right = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_right, [self.som_dim[0] * self.som_dim[1], 1])))
        q_left = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_left, [self.som_dim[0] * self.som_dim[1], 1])))

        q_neighbours = tf.concat([tf.expand_dims(q_up, -1), tf.expand_dims(q_down, -1),
                                  tf.expand_dims(q_right, -1), tf.expand_dims(q_left, -1)], axis=2)
        q_neighbours = tf.reduce_sum(tf.math.log(q_neighbours), axis=-1)

        #mask = tf.greater(self.q, 0.05 * tf.ones_like(self.q))
        #new_q = tf.multiply(self.q, tf.cast(mask, tf.float32))
        new_q = self.q
        q_n = tf.math.multiply(q_neighbours, tf.stop_gradient(new_q))
        q_n = tf.reduce_sum(q_n, axis=-1)
        qq = tf.math.negative(tf.reduce_mean(q_n))
        return qq

    @lazy_scope
    def loss_prediction(self):
        """Compute the prediction loss"""
        z_e = tf.reshape(self.z_e_sample, [self.batch_size, self.step_size, self.latent_dim])
        z_e_next = tf.concat([z_e[:, 1:], tf.reshape(z_e[:, -1], [-1, 1, self.latent_dim])], axis=1)
        z_e_next = tf.stop_gradient(tf.reshape(z_e_next, [-1, self.latent_dim]))
        loss_prediction = - tf.reduce_mean(self.prediction.log_prob(z_e_next))
        #loss_prediction = tf.reduce_mean(
        #    tf.squared_difference(tf.stop_gradient(tf.reshape(z_e_next, [-1, self.latent_dim])),
        #                          self.prediction))
        return loss_prediction

    @lazy_scope
    def loss_smoothness(self):
        """Compute the smoothness loss"""
        k_reshaped = tf.reshape(self.k, [self.batch_size, self.step_size])
        k_old = tf.concat([k_reshaped[:, 0:1], k_reshaped[:, :-1]], axis=1)
        k_old = tf.reshape(tf.cast(k_old, tf.int64), [-1, 1])
        emb = tf.reshape(self.embeddings, [self.som_dim[0] * self.som_dim[1], self.latent_dim])
        e = tf.gather_nd(emb, k_old)
        diff = tf.reduce_sum(tf.squared_difference(self.z_e_sample, tf.stop_gradient(e)), axis=-1)
        q = tf.keras.backend.epsilon() + (1.0 / (1.0 + diff/self.alpha) ** ((self.alpha + 1.0) / 2.0))
        loss_smoothness = - tf.reduce_mean(q)
        return loss_smoothness

    @lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss."""
        loss = self.theta * self.loss_reconstruction_ze + self.gamma * self.loss_commit + self.beta * self.loss_som + \
                self.kappa * self.loss_smoothness + self.eta*self.loss_prediction
        tf.summary.scalar("loss_rec", self.theta * self.loss_reconstruction_ze)
        tf.summary.scalar("loss_commit",  self.gamma * self.loss_commit)
        tf.summary.scalar("loss_som", self.beta * self.loss_som)
        tf.summary.scalar("loss_smoothness", self.kappa * self.loss_smoothness)
        tf.summary.scalar("loss_prediction", self.eta*self.loss_prediction)
        tf.summary.scalar("loss", loss)
        return loss

    @lazy_scope
    def loss_commit_sd(self):
        """Computes the commitment loss of standard SOM for initialization."""
        loss_commit = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.z_e_sample), self.z_q))
        tf.summary.scalar("loss_commit_sd", loss_commit)
        return loss_commit

    @lazy_scope
    def loss_som_old(self):
        """Computes the SOM loss."""
        loss_som = tf.reduce_mean(
            tf.squared_difference(tf.expand_dims(tf.stop_gradient(self.z_e_sample), axis=1), self.z_q_neighbors))
        tf.summary.scalar("loss_som_old", loss_som)
        return loss_som

    @lazy_scope
    def loss_a(self):
        """Aggregates the loss terms into the total loss."""
        loss = self.loss_som_old + self.loss_commit_sd
        tf.summary.scalar("loss_pre", loss)
        return loss

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor,
                                              staircase=True)
        optimizer = tf.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        train_step_prob = optimizer.minimize(self.loss_prediction, global_step=self.global_step)
        train_step_ae = optimizer.minimize(self.loss_reconstruction_ze, global_step=self.global_step)
        train_step_som = optimizer.minimize(self.loss_a, global_step=self.global_step)
        return train_step, train_step_ae, train_step_som, train_step_prob
