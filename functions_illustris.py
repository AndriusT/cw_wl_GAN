"""
Functions for training the cosmoGAN algorithm and to analyze the results for
the Illustris data. The main difference here is that the dataset arrays in the
training loop are multidimensional rather than 1-D.
"""


###################################
##    Importing the packages:    ##
###################################

import os
#os.chdir('/users/tamosiua/cosmoGAN/cosmoGAN/networks/models')
#os.chdir('./models')
import tensorflow as tf
import sys
#print(sys.path)
#from ops import linear, conv2d, conv2d_transpose, lrelu
import time
import numpy as np
import pprint

###########################
## Checkpoint settings:  ##
###########################


def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, transpose_b=False):

    shape = input_.get_shape().as_list()
    if not transpose_b:
        w_shape = [shape[1], output_size]
    else:
        w_shape = [output_size, shape[1]]

    with tf.variable_scope(scope or "linear"):
        matrix = tf.get_variable('w', w_shape, tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable('b', [output_size],
            initializer=tf.constant_initializer(bias_start))

        return tf.matmul(input_, matrix, transpose_b=transpose_b) + bias

def conv2d(input_, out_channels, data_format, kernel=5, stride=2, stddev=0.02, name="conv2d"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        strides = [1, stride, stride, 1]
    else: # NCHW
        in_channels = input_.get_shape()[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, in_channels, out_channels],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=strides, padding='SAME', data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases, data_format=data_format), conv.get_shape())

        return conv

def conv2d_transpose(input_, output_shape, data_format, kernel=5, stride=2, stddev=0.02,
                     name="conv2d_transpose"):

    if data_format == "NHWC":
        in_channels = input_.get_shape()[-1]
        out_channels = output_shape[-1]
        strides = [1, stride, stride, 1]
    else:
        in_channels = input_.get_shape()[1]
        out_channels = output_shape[1]
        strides = [1, 1, stride, stride]

    with tf.variable_scope(name):
        w = tf.get_variable('w', [kernel, kernel, out_channels, in_channels],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=strides, data_format=data_format)

        biases = tf.get_variable('biases', [out_channels], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases, data_format=data_format), deconv.get_shape())

        return deconv


def lrelu(x, alpha=0.2, name="lrelu"):
    with tf.name_scope(name):
      return tf.maximum(x, alpha*x)

class dcgan(object):
    def __init__(self, output_size=16, batch_size=64,
                 nd_layers=4, ng_layers=4, df_dim=128, gf_dim=128,
                 c_dim=3, z_dim=4, flip_labels=0.01, data_format="NHWC",
                 gen_prior=tf.random_normal, transpose_b=False):

        self.output_size = output_size
        self.batch_size = batch_size
        self.nd_layers = nd_layers
        self.ng_layers = ng_layers
        self.df_dim = df_dim
        self.gf_dim = gf_dim
        self.c_dim = c_dim
        self.z_dim = z_dim
        self.flip_labels = flip_labels
        self.data_format = data_format
        self.gen_prior = gen_prior
        self.transpose_b = transpose_b # transpose weight matrix in linear layers for (possible) better performance when running on HSW/KNL
        self.stride = 2 # this is fixed for this architecture

        self._check_architecture_consistency()


        self.batchnorm_kwargs = {'epsilon' : 1e-5, 'decay': 0.9,
                                 'updates_collections': None, 'scale': True,
                                 'fused': True, 'data_format': self.data_format}

    def training_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = self.gen_prior(shape=[self.batch_size, self.z_dim])

        with tf.variable_scope("discriminator") as d_scope:
            d_prob_real, d_logits_real = self.discriminator(self.images, is_training=True)

        with tf.variable_scope("generator") as g_scope:
            g_images = self.generator(self.z, is_training=True)

        with tf.variable_scope("discriminator") as d_scope:
            d_scope.reuse_variables()
            d_prob_fake, d_logits_fake = self.discriminator(g_images, is_training=True)

        with tf.name_scope("losses"):
            with tf.name_scope("d"):
                d_label_real, d_label_fake = self._labels()
                self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=d_label_real, name="real"))
                self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=d_label_fake, name="fake"))
                self.d_loss = self.d_loss_real + self.d_loss_fake
            with tf.name_scope("g"):
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_logits_fake)))

        self.d_summary = tf.summary.merge([tf.summary.histogram("prob/real", d_prob_real),
                                           tf.summary.histogram("prob/fake", d_prob_fake),
                                           tf.summary.scalar("loss/real", self.d_loss_real),
                                           tf.summary.scalar("loss/fake", self.d_loss_fake),
                                           tf.summary.scalar("loss/d", self.d_loss)])

        g_sum = [tf.summary.scalar("loss/g", self.g_loss)]
        if self.data_format == "NHWC": # tf.summary.image is not implemented for NCHW
            g_sum.append(tf.summary.image("G", g_images, max_outputs=4))
        self.g_summary = tf.summary.merge(g_sum)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'discriminator/' in var.name]
        self.g_vars = [var for var in t_vars if 'generator/' in var.name]

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

    def inference_graph(self):

        if self.data_format == "NHWC":
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.output_size, self.output_size, self.c_dim], name='real_images')
        else:
            self.images = tf.placeholder(tf.float32, [self.batch_size, self.c_dim, self.output_size, self.output_size], name='real_images')

        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')

        with tf.variable_scope("discriminator") as d_scope:
            self.D,_ = self.discriminator(self.images, is_training=False)

        with tf.variable_scope("generator") as g_scope:
            self.G = self.generator(self.z, is_training=False)

        with tf.variable_scope("counters") as counters_scope:
            self.epoch = tf.Variable(-1, name='epoch', trainable=False)
            self.increment_epoch = tf.assign(self.epoch, self.epoch+1)
            self.global_step = tf.train.get_or_create_global_step()

        self.saver = tf.train.Saver(max_to_keep=8000)

    def optimizer(self, learning_rate, beta1):

        d_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.d_loss, var_list=self.d_vars, global_step=self.global_step)

        g_optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
                                         .minimize(self.g_loss, var_list=self.g_vars)

        return tf.group(d_optim, g_optim, name="all_optims")


    def generator(self, z, is_training):

        map_size = self.output_size/int(2**self.ng_layers)
        num_channels = self.gf_dim * int(2**(self.ng_layers -1))

        # h0 = relu(BN(reshape(FC(z))))
        z_ = linear(z, num_channels*map_size*map_size, 'h0_lin', transpose_b=self.transpose_b)
        h0 = tf.reshape(z_, self._tensor_data_format(-1, map_size, map_size, num_channels))
        bn0 = tf.contrib.layers.batch_norm(h0, is_training=is_training, scope='bn0', **self.batchnorm_kwargs)
        h0 = tf.nn.relu(bn0)

        chain = h0
        for h in range(1, self.ng_layers):
            # h1 = relu(BN(conv2d_transpose(h0)))
            map_size *= self.stride
            num_channels /= 2
            chain = conv2d_transpose(chain,
                                     self._tensor_data_format(self.batch_size, map_size, map_size, num_channels),
                                     stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%h)
            chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
            chain = tf.nn.relu(chain)

        # h1 = conv2d_transpose(h0)
        map_size *= self.stride
        hn = conv2d_transpose(chain,
                              self._tensor_data_format(self.batch_size, map_size, map_size, self.c_dim),
                              stride=self.stride, data_format=self.data_format, name='h%i_conv2d_T'%(self.ng_layers))

        return tf.nn.tanh(hn)


    def discriminator(self, image, is_training):

        # h0 = lrelu(conv2d(image))
        h0 = lrelu(conv2d(image, self.df_dim, self.data_format, name='h0_conv'))

        chain = h0
        for h in range(1, self.nd_layers):
            # h1 = lrelu(BN(conv2d(h0)))
            chain = conv2d(chain, self.df_dim*(2**h), self.data_format, name='h%i_conv'%h)
            chain = tf.contrib.layers.batch_norm(chain, is_training=is_training, scope='bn%i'%h, **self.batchnorm_kwargs)
            chain = lrelu(chain)

        # h1 = linear(reshape(h0))
        hn = linear(tf.reshape(chain, [self.batch_size, -1]), 1, 'h%i_lin'%self.nd_layers, transpose_b=self.transpose_b)

        return tf.nn.sigmoid(hn), hn

    def _labels(self):
        with tf.name_scope("labels"):
            ones = tf.ones([self.batch_size, 1])
            zeros = tf.zeros([self.batch_size, 1])
            flip_labels = tf.constant(self.flip_labels)

            if self.flip_labels > 0:
                prob = tf.random_uniform([], 0, 1)

                d_label_real = tf.cond(tf.less(prob, flip_labels), lambda: zeros, lambda: ones)
                d_label_fake = tf.cond(tf.less(prob, flip_labels), lambda: ones, lambda: zeros)
            else:
                d_label_real = ones
                d_label_fake = zeros

        return d_label_real, d_label_fake

    def _tensor_data_format(self, N, H, W, C):
        if self.data_format == "NHWC":
            return [int(N), int(H), int(W), int(C)]
        else:
            return [int(N), int(C), int(H), int(W)]

    def _check_architecture_consistency(self):

        if self.output_size/2**self.nd_layers < 1:
            print("Error: Number of discriminator conv. layers are larger than the output_size for this architecture")
            exit(0)

        if self.output_size/2**self.ng_layers < 1:
            print("Error: Number of generator conv_transpose layers are larger than the output_size for this architecture")
            exit(0)

def save_checkpoint(sess, saver, tag, checkpoint_dir, counter, step=False):

    model_name = tag + '.model-'
    if step:
        model_name += 'step'
    else:
        model_name += 'epoch'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)

def load_checkpoint(sess, saver, tag, checkpoint_dir, counter=None, step=False):
    print(" [*] Reading checkpoints...")

    if step:
        counter_name = 'step'
    else:
        counter_name = 'epoch'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        if not counter==None:
            ckpt_name_epoch = ckpt_name[:ckpt_name.find(counter_name)] + counter_name + '-%i'%counter
            if os.path.exists(os.path.join(checkpoint_dir, ckpt_name_epoch+'.index')):
                ckpt_name = ckpt_name_epoch
            else:
                print("Checkpoint for ", counter_name , counter_name, "doesn't exist. Using latest checkpoint instead!")

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False

def train_dcgan(data, config):

    training_graph = tf.Graph()

    with training_graph.as_default():

        gan = dcgan(output_size=config.output_size,
                          batch_size=config.batch_size,
                          nd_layers=config.nd_layers,
                          ng_layers=config.ng_layers,
                          df_dim=config.df_dim,
                          gf_dim=config.gf_dim,
                          c_dim=config.c_dim,
                          z_dim=config.z_dim,
                          flip_labels=config.flip_labels,
                          data_format=config.data_format,
                          transpose_b=config.transpose_matmul_b)

        save_every_step = True if config.save_every_step == 'True' else False

        gan.training_graph()
        update_op = gan.optimizer(config.learning_rate, config.beta1)

        checkpoint_dir = os.path.join(config.checkpoint_dir, config.experiment)

        with tf.Session() as sess:
            writer = tf.summary.FileWriter('./logs/'+config.experiment+'/train', sess.graph)

            init_op = tf.global_variables_initializer()
            sess.run(init_op)

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, step=save_every_step)

            epoch = sess.run(gan.increment_epoch)
            start_time = time.time()
            for epoch in range(epoch, epoch + config.epoch):

                perm = np.random.permutation(data.shape[0])
                num_batches = data.shape[0] // config.batch_size

                for idx in range(0, num_batches):
                    batch_images = data[perm[idx*config.batch_size:(idx+1)*config.batch_size]]

                    _, g_sum, d_sum = sess.run([update_op, gan.g_summary, gan.d_summary],
                                               feed_dict={gan.images: batch_images})

                    global_step = gan.global_step.eval()
                    writer.add_summary(g_sum, global_step)
                    writer.add_summary(d_sum, global_step)

                    ## Uncomment this (and comment out the lines below) to save every step:
                    #save_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, global_step, step=True)

                    if save_every_step:
                        save_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, global_step, step=True)

                    if config.verbose:
                        errD_fake = gan.d_loss_fake.eval()
                        errD_real = gan.d_loss_real.eval({gan.images: batch_images})
                        errG = gan.g_loss.eval()

                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                                % (epoch, idx, num_batches, time.time() - start_time, errD_fake+errD_real, errG))

                    elif global_step%100 == 0:
                        print("Epoch: [%2d] Step: [%4d/%4d] time: %4.4f"%(epoch, idx, num_batches, time.time() - start_time))

                # save a checkpoint every epoch
		## Save only every 100th epoch (uncomment if it breaks):
                if epoch % 100 == 0:
                    save_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir, epoch, step=True)

                sess.run(gan.increment_epoch)


def get_data():
    data = np.load(config.datafile, mmap_mode='r')

#    if config.data_format == 'NHWC':
#        data = np.expand_dims(data, axis=-1)
#    else: # 'NCHW'
#        data = np.expand_dims(data, axis=1)

    return data


def save_checkpoint(sess, saver, tag, checkpoint_dir, counter, step=True):

    model_name = tag + '.model-'
    if step:
        model_name += 'step'
    else:
        model_name += 'epoch'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    saver.save(sess, os.path.join(checkpoint_dir, model_name), global_step=counter)

def load_checkpoint(sess, saver, tag, checkpoint_dir, counter=None, step=True):
    print(" [*] Reading checkpoints...")

    if step:
        counter_name = 'step'
    else:
        counter_name = 'epoch'

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        if not counter==None:
            ckpt_name_epoch = ckpt_name[:ckpt_name.find(counter_name)] + counter_name + '-%i'%counter
            if os.path.exists(os.path.join(checkpoint_dir, ckpt_name_epoch+'.index')):
                ckpt_name = ckpt_name_epoch
            else:
                print("Checkpoint for ", counter_name , counter_name, "doesn't exist. Using latest checkpoint instead!")

        saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
        print(" [*] Success to read {}".format(ckpt_name))
        return True
    else:
        print(" [*] Failed to find a checkpoint")
        return False

def Loader(checkpoint_dir_pt ,checkpoint_epoch, output_size = 256, z = 256):
    '''
    Initiates a tensorflow session and loads the input checkpoint at a given epoch.
    Also produces a batch of samples corresponding to the newest.
    '''
    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            gan = dcgan(output_size=output_size,
                          nd_layers=4,
                          ng_layers=4,
                          df_dim=64,
                          gf_dim=64,
                          z_dim=z,
                          data_format="NHWC")

            gan.inference_graph()

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir_pt, counter=checkpoint_epoch)

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator/' in var.name]
            g_vars = [var for var in t_vars if 'generator/' in var.name]

            g_vars = [(sess.run(var), var.name) for var in g_vars]
            d_vars = [(sess.run(var), var.name) for var in d_vars]


    with tf.Graph().as_default() as g:
        with tf.Session(graph=g) as sess:

            gan = dcgan(output_size=output_size,
                          nd_layers=4,
                          ng_layers=4,
                          df_dim=64,
                          gf_dim=64,
                          z_dim=z,
                          data_format="NHWC")

            gan.inference_graph()

            load_checkpoint(sess, gan.saver, 'dcgan', checkpoint_dir_pt, counter=checkpoint_epoch)

            z_sample = np.random.normal(size=(gan.batch_size, gan.z_dim))
            #z_sample = np.random.normal(size=(batch_size, gan.z_dim))
            samples = sess.run(gan.G, feed_dict={gan.z: z_sample})

            t_vars = tf.trainable_variables()
            d_vars = [var for var in t_vars if 'discriminator/' in var.name]
            g_vars = [var for var in t_vars if 'generator/' in var.name]

            g_vars = [(sess.run(var), var.name) for var in g_vars]# if 'g_h' in var.name]
            d_vars = [(sess.run(var), var.name) for var in d_vars]#if 'd_h' in var.name]

    return z_sample, samples
