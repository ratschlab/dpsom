"""
Model architecture of the DPSOM model.
"""

import functools

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

class DPSOM:
    """Class for the DPSOM model"""

    def __init__(self, latent_dim=100, som_dim=[8,8], learning_rate=1e-4, decay_factor=0.99, decay_steps=1000,
                 input_length=28, input_channels=28, alpha=10., beta=20., gamma=20., theta=1., dropout=0.5, prior_var=1,
                 prior=0.5, convolution=False):
        """Initialization method for the DPSOM model object.
        Args:
            latent_dim (int): The dimensionality of the latent embeddings (default: 100).
            som_dim (list): The dimensionality of the self-organizing map (default: [8,8]).
            learning_rate (float): The learning rate for the optimization (default: 1e-4).
            decay_factor (float): The factor for the learning rate decay (default: 0.95).
            decay_steps (int): The number of optimization steps before every learning rate
                decay (default: 1000).
            input_length (int): The length of the input data points (default: 28).
            input_channels (int): The number of channels of the input data points (default: 28).
            alpha (float): The weight for the commitment loss (default: 10.).
            beta (float): Weight for the SOM loss (default: 20).
            gamma (float): Weight for the KL term of the PSOM clustering loss (default: 20).
            theta (float): Weight for the VAE loss (default: 1).
            dropout (float): Dropout factor for the feed-forward layers of the VAE (default: 0.5).
            prior_var (float): Multiplier of the diagonal variance of the VAE multivariate gaussian prior (default: 1).
            prior (float): Weight of the regularization term of the ELBO (default: 0.5).
            convolution (bool): Indicator if the model use convolutional layers (True) or feed-forward layers (False)
                                (default: False).
        """
        self.latent_dim = latent_dim
        self.som_dim = som_dim
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.input_length = input_length
        self.input_channels = input_channels
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.theta = theta
        self.dropout = dropout
        self.prior_var = prior_var
        self.prior = prior
        self.convolution = convolution
        self.is_training
        self.input_dim
        self.inputs
        self.batch_size
        self.embeddings
        self.global_step
        self.z_e
        self.sample_z_e
        self.z_dist_flat
        self.k
        self.z_q
        self.z_q_neighbors
        self.reconstruction_e
        self.loss_reconstruction_ze
        self.p
        self.q
        self.loss_commit
        self.loss_commit_s
        self.loss_som_s
        self.loss_som
        self.loss
        self.loss_a
        self.optimize

    @lazy_scope
    def is_training(self):
        is_training = tf.placeholder(tf.bool, shape=[], name="is_training")
        return is_training

    @lazy_scope
    def input_dim(self):
        x = 28*28
        return x

    @lazy_scope
    def inputs(self):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name="x")
        return x

    @lazy_scope
    def batch_size(self):
        """Reads the batch size from the input tensor."""
        batch_size = tf.shape(self.inputs)[0]
        return batch_size

    @lazy_scope
    def embeddings(self):
        """Creates variable for the SOM embeddings."""
        embeddings = tf.get_variable("embeddings", self.som_dim+[self.latent_dim],
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
            if not self.convolution:
                inputs = tf.reshape(self.inputs, (-1, self.input_dim))
                dense1 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(inputs)
                dense1 = tf.keras.layers.Dropout(rate=self.dropout)(dense1)
                dense1 = BatchNormalization()(dense1)
                dense2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense1)
                dense2 = tf.keras.layers.Dropout(rate=self.dropout)(dense2)
                dense2 = BatchNormalization()(dense2)
                dense3 = tf.keras.layers.Dense(2000, activation=tf.nn.leaky_relu)(dense2)
                dense3 = tf.keras.layers.Dropout(rate=self.dropout)(dense3)
                flattened = BatchNormalization()(dense3)
                #flattened = dense3
            else:
                inputs = tf.cast(self.inputs, tf.float32) - 0.5
                conv1 = Conv2D(32, (3, 3), activation=tf.nn.leaky_relu, padding='same')(inputs)  # 28 x 28 x 32
                conv1 = BatchNormalization()(conv1)
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
                conv2 = Conv2D(64, (3, 3), activation=tf.nn.leaky_relu, padding='same')(pool1)  # 14 x 14 x 64
                conv2 = BatchNormalization()(conv2)
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
                conv3 = Conv2D(128, (3, 3), activation=tf.nn.leaky_relu, padding='same')(
                    pool2)
                conv3 = BatchNormalization()(conv3)
                conv4 = Conv2D(256, (3, 3), activation=tf.nn.leaky_relu, padding='same')(
                    conv3)
                conv4 = BatchNormalization()(conv4)
                flattened = tf.reshape(conv4, [-1, 7 * 7 * 256])

            z_e = Dense(tfp.layers.MultivariateNormalTriL.params_size(self.latent_dim), activation=None)(flattened)
            #prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.latent_dim), scale=1),
            #                                      reinterpreted_batch_ndims=1)
            z_e = tfp.layers.MultivariateNormalTriL(self.latent_dim)(z_e)
                                                    #activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior, weight=self.prior))(z_e)
        return z_e

    @lazy_scope
    def sample_z_e(self):
        """Sample from the distribution of probability of the latent embeddings."""
        sample_z_e = tf.identity(self.z_e, name="z_e")
        return sample_z_e

    @lazy_scope
    def z_dist_flat(self):
        """Computes the distances between the centroids and the embeddings."""
        z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(self.sample_z_e, 1), 1), tf.expand_dims(self.embeddings, 0))
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[1]])
        return z_dist_flat

    @lazy_scope
    def z_dist_flat_ng(self):
        """Computes the distances between the centroids and the embeddings stopping the gradient of the latent
        embeddings."""
        z_dist = tf.squared_difference(tf.expand_dims(tf.expand_dims(tf.stop_gradient(self.sample_z_e), 1), 1),
                                       tf.expand_dims(self.embeddings, 0))
        z_dist_red = tf.reduce_sum(z_dist, axis=-1)  # 1,32,8,8
        z_dist_flat = tf.reshape(z_dist_red, [-1, self.som_dim[0] * self.som_dim[1]])  # 1,32,64
        return z_dist_flat

    @lazy_scope
    def k(self):
        """Picks the index of the closest centroid for every embedding."""
        k = tf.argmin(self.z_dist_flat, axis=-1, name="k")
        return k

    @lazy_scope
    def z_q(self):
        """Aggregates the respective closest centroids for every embedding."""
        k_1 = self.k // self.som_dim[0]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)
        z_q = tf.gather_nd(self.embeddings, k_stacked, name="z_q")
        return z_q

    @lazy_scope
    def z_q_neighbors(self):
        """Aggregates the respective neighbors in the SOM grid for every z_q."""
        k_1 = self.k // self.som_dim[0]
        k_2 = self.k % self.som_dim[1]
        k_stacked = tf.stack([k_1, k_2], axis=1)

        k1_not_top = tf.less(k_1, tf.constant(self.som_dim[0]-1, dtype=tf.int64))
        k1_not_bottom = tf.greater(k_1, tf.constant(0, dtype=tf.int64))
        k2_not_right = tf.less(k_2, tf.constant(self.som_dim[1]-1, dtype=tf.int64))
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
        """Reconstructs the input from the encodings by learning a Bernoulli distribution."""
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            z_p = tf.placeholder(tf.float32, shape=[None, self.latent_dim], name="z_e")
            z_e = tf.cond(self.is_training, lambda: self.z_e, lambda: z_p)
            if not self.convolution:
                flat_size = 2000
                dense4 = tf.keras.layers.Dense(flat_size, activation=tf.nn.leaky_relu)(z_e)
                dense4 = BatchNormalization()(dense4)
                dense3 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense4)
                dense3 = BatchNormalization()(dense3)
                dense2 = tf.keras.layers.Dense(500, activation=tf.nn.leaky_relu)(dense3)
                dense1 = BatchNormalization()(dense2)
                x_hat = tf.keras.layers.Dense(self.input_dim, activation=tf.nn.leaky_relu)(dense1)
            else:
                flat_size = 7 * 7 * 256
                dense3 = tf.keras.layers.Dense(flat_size, activation=tf.nn.leaky_relu)(z_e)
                h_reshaped = tf.reshape(dense3, [-1, 7, 7, 256])
                conv5 = Conv2DTranspose(128, (3, 3), activation=tf.nn.leaky_relu, padding='same')(
                    h_reshaped)  # 7 x 7 x 128
                conv5 = BatchNormalization()(conv5)
                conv6 = Conv2DTranspose(64, (3, 3), activation=tf.nn.leaky_relu, padding='same')(conv5)  # 7 x 7 x 64
                conv6 = BatchNormalization()(conv6)
                up1 = UpSampling2D((2, 2))(conv6)  # 14 x 14 x 64
                conv7 = Conv2DTranspose(32, (3, 3), activation=tf.nn.leaky_relu, padding='same')(up1)  # 14 x 14 x 32
                conv7 = BatchNormalization()(conv7)
                up2 = UpSampling2D((2, 2))(conv7)  # 28 x 28 x 32
                x_hat = Conv2D(1, (3, 3), activation=None, padding='same')(up2)  # 28 x 28 x 1
                x_hat = tf.reshape(x_hat, [-1, 28 * 28])
            x_hat = tfp.layers.IndependentBernoulli([28, 28, 1], tfp.distributions.Bernoulli.logits)(x_hat)
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
        loss_rec = log_lik_loss + self.prior*kl_loss
        return loss_rec

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
            q = tf.keras.backend.epsilon() + 1.0 / (1.0 + self.z_dist_flat_ng / self.alpha) ** (
                        (self.alpha + 1.0) / 2.0)
            q = (q / tf.reduce_sum(q, axis=1, keepdims=True))
        return q

    @lazy_scope
    def p(self):
        """Placeholder for the target distribution."""
        p = tf.placeholder(tf.float32, shape=(None, self.som_dim[0]*self.som_dim[1]))
        return p

    @lazy_scope
    def loss_commit(self):
        """Computes the KL term of the clustering loss."""
        loss_commit = tf.reduce_mean(tf.reduce_sum(self.p * tf.log(self.p / self.q), axis=1))
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
        k1_down = tf.where(k1_not_bottom, tf.subtract(k_1, 1),
                           tf.ones(tf.shape(k_1), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1))
        k2_right = tf.where(k2_not_right, tf.add(k_2, 1), tf.zeros(tf.shape(k_2), dtype=tf.dtypes.int32))
        k2_left = tf.where(k2_not_left, tf.subtract(k_2, 1),
                           tf.ones(tf.shape(k_2), dtype=tf.dtypes.int32) * (self.som_dim[0] - 1))

        k_up = k1_up * self.som_dim[0] + k_2
        k_down = k1_down * self.som_dim[0] + k_2
        k_right = k_1 * self.som_dim[0] + k2_right
        k_left = k_1 * self.som_dim[0] + k2_left

        q_t = tf.transpose(self.q_ng)
        q_up = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_up, [self.som_dim[0] * self.som_dim[1], 1])))
        q_down = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_down, [self.som_dim[0] * self.som_dim[1], 1])))
        q_right = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_right, [self.som_dim[0] * self.som_dim[1], 1])))
        q_left = tf.transpose(tf.gather_nd(q_t, tf.reshape(k_left, [self.som_dim[0] * self.som_dim[1], 1])))
        q_neighbours = tf.stack([q_up, q_down, q_right, q_left], axis=2)
        q_neighbours = tf.reduce_sum(tf.math.log(q_neighbours), axis=-1)
        # threshold
        #maxx = 0.1
        #mask = tf.greater_equal(self.q, maxx * tf.ones_like(self.q))
        #new_q = tf.multiply(self.q, tf.cast(mask, tf.float32))
        new_q = self.q
        q_n = tf.math.multiply(q_neighbours, tf.stop_gradient(new_q))
        q_n = tf.reduce_sum(q_n, axis=-1)
        qq = - tf.reduce_mean(q_n)
        return qq

    @lazy_scope
    def loss(self):
        """Aggregates the loss terms into the total loss."""
        loss = (self.theta*self.loss_reconstruction_ze+self.gamma*self.loss_commit + self.beta*self.loss_som)
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("loss_som", self.beta*self.loss_som)
        tf.summary.scalar("loss_commit", self.gamma*self.loss_commit)
        tf.summary.scalar("loss_elbo", self.theta*self.loss_reconstruction_ze)
        return loss

    @lazy_scope
    def loss_commit_s(self):
        """Computes the commitment loss of standard SOM for initialization."""
        loss_commit = tf.reduce_mean(tf.squared_difference(tf.stop_gradient(self.sample_z_e), self.z_q))
        tf.summary.scalar("loss_commit_s", loss_commit)
        return loss_commit

    @lazy_scope
    def loss_som_s(self):
        """Computes the SOM loss of standard SOM for initialization."""
        loss_som = tf.reduce_mean(tf.squared_difference(tf.expand_dims(tf.stop_gradient(self.sample_z_e), axis=1), self.z_q_neighbors))
        tf.summary.scalar("loss_som_s", loss_som)
        return loss_som

    @lazy_scope
    def loss_a(self):
        """Clustering loss of standard SOM used for initialization."""
        loss = self.loss_som_s + self.loss_commit_s
        return loss

    @lazy_scope
    def optimize(self):
        """Optimizes the model's loss using Adam with exponential learning rate decay."""
        lr_decay = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_factor, staircase=True)
        optimizer = tf.train.AdamOptimizer(lr_decay)
        train_step = optimizer.minimize(self.loss, global_step=self.global_step)
        train_step_ae = optimizer.minimize(self.loss_reconstruction_ze, global_step=self.global_step)
        train_step_som = optimizer.minimize(self.loss_a, global_step=self.global_step)
        return train_step, train_step_ae, train_step_som
