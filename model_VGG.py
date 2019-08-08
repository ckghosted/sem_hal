import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from sklearn.metrics import accuracy_score

from ops import *
from utils import *

import pickle
import tqdm

import imgaug as ia
from imgaug import augmenters as iaa

VGG_MEAN = [103.939, 116.779, 123.68]
sometimes = lambda aug: iaa.Sometimes(0.5, aug)

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# VGG16 for extract features
class VGG(object):
    def __init__(self,
                 sess,
                 model_name='VGG',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 img_size_h=32,
                 img_size_w=32,
                 c_dim=3,
                 fc_dim=512,
                 n_fine_class=80,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001,
                 lambda_center_loss=0.0,
                 alpha_center_loss=0.5,
                 vgg16_npy_path='/data/put_data/cclin/ntu/dlcv2018/hw3/vgg16.npy'):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.img_size_h = img_size_h
        self.img_size_w = img_size_w
        self.c_dim = c_dim
        self.fc_dim = fc_dim
        self.n_fine_class = n_fine_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
        self.lambda_center_loss = lambda_center_loss
        self.alpha_center_loss = alpha_center_loss
        self.vgg16_weights = np.load(vgg16_npy_path, encoding='latin1').item()
        print("vgg16.npy loaded")
    
    def build_model(self):
        image_dims = [self.img_size_h, self.img_size_w, self.c_dim] ### [32,32,3] for cifar-100, [64,64,3] for mini-imagenet
        self.images = tf.placeholder(tf.float32, shape=[None]+image_dims, name='images')

        print("RGB to BGR")
        # rgb_scaled = rgb * 255.0
        ### Input layer: convert RGB to BGR and subtract pixels mean
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.images)
        self.bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])

        self.coarse_labels = tf.placeholder(tf.float32, shape=[None], name='coarse_labels')
        # self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        self.fine_labels = tf.placeholder(tf.int32, shape=[None], name='fine_labels')
        self.fine_labels_vec = tf.one_hot(self.fine_labels, self.n_fine_class)
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ## batch normalization
        self.bn_conv1_1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv1_1')
        self.bn_conv1_2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv1_2')
        self.bn_conv2_1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv2_1')
        self.bn_conv2_2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv2_2')
        self.bn_conv3_1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv3_1')
        self.bn_conv3_2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv3_2')
        self.bn_conv3_3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv3_3')
        self.bn_conv4_1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv4_1')
        self.bn_conv4_2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv4_2')
        self.bn_conv4_3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv4_3')
        self.bn_conv5_1 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv5_1')
        self.bn_conv5_2 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv5_2')
        self.bn_conv5_3 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_conv5_3')
        self.bn_dense14 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14')
        self.bn_dense15 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15')
        ### [2019/01/18] Work-around for AwA and mini-imagenet that have larger image sizes (e.g., 64-by-64)
        self.bn_dense_one_more = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense_one_more')
        
        print("build model started")
        self.features = self.build_cnn(self.bgr)
        self.logits = self.build_mlp(self.features)
        
        print("build model finished, define loss and optimizer")
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels_vec,
                                                                           logits=self.logits,
                                                                           name='loss'))

        ### Add centor loss
        self.center_loss, self.centers, self.centers_update_op = self.get_center_loss(self.features, self.fine_labels, self.alpha_center_loss, self.n_fine_class)
        self.loss_all = self.loss + self.lambda_center_loss * self.center_loss

        #### variables
        self.all_vars = tf.global_variables()
        self.all_vars_cnn = [var for var in self.all_vars if 'cnn' in var.name]
        self.all_vars_mlp = [var for var in self.all_vars if 'mlp' in var.name]
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_cnn = [var for var in self.trainable_vars if 'cnn' in var.name]
        self.trainable_vars_mlp = [var for var in self.trainable_vars if 'mlp' in var.name]
        
        #### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_mlp = [reg for reg in self.all_regs if \
                              ('Matrix' in reg.name) or ('bias' in reg.name)]
        
        #### optimizers
        with tf.control_dependencies([self.centers_update_op]):
            self.opt_mlp = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_all+sum(self.used_regs_mlp),
                                                                      var_list=self.trainable_vars_mlp)
            self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_all+sum(self.used_regs),
                                                                      var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_cnn = tf.train.Saver(var_list = self.all_vars_cnn,
                                        max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]
        
    def build_cnn(self, input_, reuse=False):
        with tf.variable_scope('cnn', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            self.conv1_1 = self.bn_conv1_1(self.conv_layer(input_, "conv1_1"), train=self.bn_train) ## [-1,32,32,64] for cifar-100, [-1,64,64,64] for mini-imagenet
            self.relu1_1 = tf.nn.relu(self.conv1_1, name='relu1_1')
            self.conv1_2 = self.bn_conv1_2(self.conv_layer(self.relu1_1, "conv1_2"), train=self.bn_train) ## [-1,32,32,64] for cifar-100, [-1,64,64,64] for mini-imagenet
            self.relu1_2 = tf.nn.relu(self.conv1_2, name='relu1_2')
            self.pool1 = self.max_pool(self.relu1_2, 'pool1')  ## [-1,16,16,64] for cifar-100, [-1,32,32,64] for mini-imagenet
            ### Layer 2
            self.conv2_1 = self.bn_conv2_1(self.conv_layer(self.pool1, "conv2_1"), train=self.bn_train) ## [-1,16,16,128] for cifar-100, [-1,32,32,128] for mini-imagenet
            self.relu2_1 = tf.nn.relu(self.conv2_1, name='relu2_1')
            self.conv2_2 = self.bn_conv2_2(self.conv_layer(self.relu2_1, "conv2_2"), train=self.bn_train) ## [-1,16,16,128] for cifar-100, [-1,32,32,128] for mini-imagenet
            self.relu2_2 = tf.nn.relu(self.conv2_2, name='relu2_2')
            self.pool2 = self.max_pool(self.relu2_2, 'pool2') ## [-1,8,8,128] for cifar-100, [-1,16,16,128] for mini-imagenet
            ### Layer 3
            self.conv3_1 = self.bn_conv3_1(self.conv_layer(self.pool2, "conv3_1"), train=self.bn_train) ## [-1,8,8,256] for cifar-100, [-1,16,16,256] for mini-imagenet
            self.relu3_1 = tf.nn.relu(self.conv3_1, name='relu3_1')
            self.conv3_2 = self.bn_conv3_2(self.conv_layer(self.relu3_1, "conv3_2"), train=self.bn_train) ## [-1,8,8,256] for cifar-100, [-1,16,16,256] for mini-imagenet
            self.relu3_2 = tf.nn.relu(self.conv3_2, name='relu3_2')
            self.conv3_3 = self.bn_conv3_3(self.conv_layer(self.relu3_2, "conv3_3"), train=self.bn_train) ## [-1,8,8,256] for cifar-100, [-1,16,16,256] for mini-imagenet
            self.relu3_3 = tf.nn.relu(self.conv3_3, name='relu3_3')
            self.pool3 = self.max_pool(self.relu3_3, 'pool3') ## [-1,4,4,256] for cifar-100, [-1,8,8,256] for mini-imagenet
            ### Layer 4
            self.conv4_1 = self.bn_conv4_1(self.conv_layer(self.pool3, "conv4_1"), train=self.bn_train) ## [-1,4,4,512] for cifar-100, [-1,8,8,512] for mini-imagenet
            self.relu4_1 = tf.nn.relu(self.conv4_1, name='relu4_1')
            self.conv4_2 = self.bn_conv4_2(self.conv_layer(self.relu4_1, "conv4_2"), train=self.bn_train) ## [-1,4,4,512] for cifar-100, [-1,8,8,512] for mini-imagenet
            self.relu4_2 = tf.nn.relu(self.conv4_2, name='relu4_2')
            self.conv4_3 = self.bn_conv4_3(self.conv_layer(self.relu4_2, "conv4_3"), train=self.bn_train) ## [-1,4,4,512] for cifar-100, [-1,8,8,512] for mini-imagenet
            self.relu4_3 = tf.nn.relu(self.conv4_3, name='relu4_3')
            self.pool4 = self.max_pool(self.relu4_3, 'pool4') ## [-1,2,2,512] for cifar-100, [-1,4,4,512] for mini-imagenet
            ### Layer 5
            self.conv5_1 = self.bn_conv5_1(self.conv_layer(self.pool4, "conv5_1"), train=self.bn_train) ## [-1,2,2,512] for cifar-100, [-1,4,4,512] for mini-imagenet
            self.relu5_1 = tf.nn.relu(self.conv5_1, name='relu5_1')
            self.conv5_2 = self.bn_conv5_2(self.conv_layer(self.relu5_1, "conv5_2"), train=self.bn_train) ## [-1,2,2,512] for cifar-100, [-1,4,4,512] for mini-imagenet
            self.relu5_2 = tf.nn.relu(self.conv5_2, name='relu5_2')
            self.conv5_3 = self.bn_conv5_3(self.conv_layer(self.relu5_2, "conv5_3"), train=self.bn_train) ## [-1,2,2,512] for cifar-100, [-1,4,4,512] for mini-imagenet
            self.relu5_3 = tf.nn.relu(self.conv5_3, name='relu5_3')
            self.pool5 = self.max_pool(self.relu5_3, 'pool5') ## [-1,1,1,512] for cifar-100, [-1,2,2,512] for mini-imagenet
            ### flatten
            # dim = tf.reduce_prod(tf.shape(self.pool5)[1:])
            # self.flattened = tf.reshape(self.pool5, [-1, self.fc_dim]) ## 512 for cifar-100, 2048 for mini-imagenet
            ### [2019/01/18] Work-around for AwA and mini-imagenet that have larger image sizes (e.g., 64-by-64)
            self.flat_dim = int((self.img_size_h / 32) * (self.img_size_w / 32) * 512)
            self.cnn_out = tf.reshape(self.pool5, [-1, self.flat_dim]) ## 512 for cifar-100, 2048 for mini-imagenet
            if int((self.img_size_h / 32)) > 1:
                self.dense_one_more = self.bn_dense_one_more(linear(self.cnn_out, self.fc_dim, name='dense_one_more'), train=self.bn_train) ## [-1,self.fc_dim]
                self.flattened = tf.nn.relu(self.dense_one_more)
            else:
                self.flattened = self.cnn_out
        return self.flattened
    
    def build_mlp(self, input_):
        with tf.variable_scope('mlp', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 14: dense with self.fc_dim neurons, BN, ReLU
            self.dense14 = self.bn_dense14(linear(input_, self.fc_dim, name='dense14'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu14 = tf.nn.relu(self.dense14, name='relu14')
            ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
            self.dense15 = self.bn_dense15(linear(self.relu14, self.fc_dim, name='dense15'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu15 = tf.nn.relu(self.dense15, name='relu15')
            ### Layer 16: dense with self.n_fine_class neurons, softmax
            self.dense16 = linear(self.relu15, self.n_fine_class, add_bias=True, name='dense16') ## [-1,self.n_fine_class]
        return self.dense16
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            return conv
    
    def get_conv_filter(self, name):
        #var=tf.Variable(self.vgg16_weights[name][0], name="filter_"+name)
        var=tf.get_variable(name="filter_"+name,
                            #shape=self.vgg16_weights[name][0].shape,
                            initializer=tf.convert_to_tensor(self.vgg16_weights[name][0], np.float32))
        return var
    
    def get_center_loss(self, features, labels, alpha, num_classes):
        len_features = features.get_shape()[1]
        centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
            initializer=tf.constant_initializer(0), trainable=False)
        labels = tf.reshape(labels, [-1])
        centers_batch = tf.gather(centers, labels)
        loss = tf.nn.l2_loss(features - centers_batch)
        
        diff = centers_batch - features
        unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
        
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff
        
        centers_update_op = tf.scatter_sub(centers, labels, diff)
        
        return loss, centers, centers_update_op

    def train(self,
              train_path='/data/put_data/cclin/datasets/cifar-100-python/train_base',
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=5e-5,
              num_epoch=50,
              patience=10,
              ratio_for_dense=0.0):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models_cnn'))
        
        ### Load training data (as a whole dictionary),
        ### and split each of them into training/validation by 80/20
        train_dict = unpickle(train_path)
        data_len = len(train_dict[b'fine_labels'])
        
        ## work-around for mini-imagenet (shuffle all data before splitting training/validation)
        arr_all = np.arange(data_len)
        np.random.shuffle(arr_all)
        train_dict[b'data'] = train_dict[b'data'][arr_all]
        train_dict[b'fine_labels'] = [train_dict[b'fine_labels'][idx] for idx in arr_all]

        data_train = train_dict[b'data'][0:int(data_len*0.8)].reshape((-1, 3, self.img_size_h, self.img_size_w)).transpose([0, 2, 3, 1])
        data_valid = train_dict[b'data'][int(data_len*0.8):int(data_len)].reshape((-1, 3, self.img_size_h, self.img_size_w)).transpose([0, 2, 3, 1])
        ## work-around for mini-imagenet (Cast data from float64 to uint8)
        if data_train.dtype != 'uint8':
            print('Cast data from %s to uint8' % data_train.dtype)
            data_train = np.rint(data_train*255).astype('uint8')
            data_valid = np.rint(data_valid*255).astype('uint8')
        nBatches = int(np.ceil(data_train.shape[0] / bsize))
        nBatches_valid = int(np.ceil(data_valid.shape[0] / bsize))
        #### one-hot, but need to consider the following error first:
        #### "IndexError: index 86 is out of bounds for axis 0 with size 80"
        #### Make a dictionary for {old_label: new_label} mapping, e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:79}
        #### such that all labels become 0~79
        label_mapping = {}
        for new_lb in range(self.n_fine_class):
            label_mapping[np.sort(list(set(train_dict[b'fine_labels'])))[new_lb]] = new_lb
        fine_labels = [int(s) for s in train_dict[b'fine_labels'][0:int(data_len*0.8)]]
        # fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels]
        # fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
        fine_labels_train = np.array([label_mapping[old_lb] for old_lb in fine_labels])
        fine_labels = [int(s) for s in train_dict[b'fine_labels'][int(data_len*0.8):int(data_len)]]
        # fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels]
        # fine_labels_valid = np.eye(self.n_fine_class)[fine_labels_new]
        fine_labels_valid = np.array([label_mapping[old_lb] for old_lb in fine_labels])
        
        ### Data indexes used to shuffle training order
        arr = np.arange(data_train.shape[0])
        
        ## Basic image augmentation
        seq = iaa.Sequential([
            iaa.Crop(px=(0, 4)), # crop images from each side by 0 to 4px (randomly chosen)
            iaa.Fliplr(0.5), # horizontally flip 50% of the images
            iaa.GaussianBlur(sigma=(0, 0.5)), # blur images with a sigma of 0 to 0.5
            sometimes(iaa.Affine(
                scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 85-115% of their size, individually per axis
                translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -15 to +15 percent (per axis)
                rotate=(-30, 30), # rotate by -30 to +30 degrees
                shear=(-12, 12), # shear by -12 to +12 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            ))
        ])
    
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        top_n_acc_train = []
        top_n_acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            loss_train_batch = []
            loss_valid_batch = []
            acc_train_batch = []
            acc_valid_batch = []
            top_n_acc_train_batch = []
            top_n_acc_valid_batch = []
            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in tqdm.tqdm(range(nBatches)):
                batch_data = data_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_data_aug = seq.augment_images(batch_data)  # done by the library
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                if epoch <= num_epoch*ratio_for_dense:
                    _, loss, logits = self.sess.run([self.opt_mlp, self.loss, self.logits],
                                                    feed_dict={self.images: batch_data_aug,
                                                               self.fine_labels: batch_labels,
                                                               self.bn_train: True,
                                                               self.keep_prob: 0.5,
                                                               self.learning_rate: learning_rate})
                else:
                    _, loss, logits = self.sess.run([self.opt_all, self.loss, self.logits],
                                                    feed_dict={self.images: batch_data_aug,
                                                               self.fine_labels: batch_labels,
                                                               self.bn_train: True,
                                                               self.keep_prob: 0.5,
                                                               self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                # y_true = np.argmax(batch_labels, axis=1)
                y_true = batch_labels
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### compute validation loss
            #print('validation')
            for idx in tqdm.tqdm(range(nBatches_valid)):
                batch_data = data_valid[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_valid[idx*bsize:(idx+1)*bsize]
                #print(batch_labels.shape)
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.images: batch_data,
                                                        self.fine_labels: batch_labels,
                                                        self.bn_train: False,
                                                        self.keep_prob: 1.0,})
                loss_valid_batch.append(loss)
                # y_true = np.argmax(batch_labels, axis=1)
                y_true = batch_labels
                y_pred = np.argmax(logits, axis=1)
                acc_valid_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_valid_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            loss_valid.append(np.mean(loss_valid_batch))
            acc_train.append(np.mean(acc_train_batch))
            acc_valid.append(np.mean(acc_valid_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            top_n_acc_valid.append(np.mean(top_n_acc_valid_batch))
            print('Epoch: %d, train loss: %f, valid loss: %f, train accuracy: %f, valid accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(loss_valid_batch), np.mean(acc_train_batch), np.mean(acc_valid_batch)))
            print('           top-%d train accuracy: %f, top-%d valid accuracy: %f' % \
                  (n_top, np.mean(top_n_acc_train_batch), n_top, np.mean(top_n_acc_valid_batch)))
            
            ### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_valid_batch)
            current_acc = np.mean(acc_valid_batch)
            if epoch == 1:
                best_loss = current_loss
                best_acc = current_acc
            else:
                #if current_loss < best_loss or current_acc > best_acc:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    best_acc = current_acc
                    self.saver.save(self.sess,
                                    os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    global_step=epoch)
                    self.saver_cnn.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_cnn', self.model_name + '.model-cnn'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]
    
    def inference(self,
                  test_path='/data/put_data/cclin/datasets/cifar-100-python/test',
                  gen_from=None, ## e.g., model_name (must given)
                  gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  out_path=None,
                  n_top=5,
                  bsize=32):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            
            #### load testing data
            test_dict = unpickle(test_path)
            ### [3072] with the first 1024 being 'R', the middle 1024 being 'G', and the last 1024 being 'B'
            ### reshape and modify rank order from (batch, channel, height, width) to (batch, height, width, channel)
            data_test = test_dict[b'data'].reshape((-1, 3, self.img_size_h, self.img_size_w)).transpose([0, 2, 3, 1])
            ## work-around for mini-imagenet
            if data_test.dtype != 'uint8':
                print('Cast data from %s to uint8' % data_test.dtype)
                data_test = np.rint(data_test*255).astype('uint8')
            nBatches_test = int(np.ceil(data_test.shape[0] / bsize))
            #### one-hot, but need to consider the following error first:
            #### "IndexError: index 86 is out of bounds for axis 0 with size 80"
            #### Make a dictionary for {old_label: new_label} mapping, e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:79}
            #### such that all labels become 0~79
            label_mapping = {}
            for new_lb in range(self.n_fine_class):
                label_mapping[np.sort(list(set(test_dict[b'fine_labels'])))[new_lb]] = new_lb
            fine_labels = [int(s) for s in test_dict[b'fine_labels']]
            # fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels]
            # fine_labels_test = np.eye(self.n_fine_class)[fine_labels_new]
            fine_labels_test = np.array([label_mapping[old_lb] for old_lb in fine_labels])
            
            ### make prediction and compute accuracy
            loss_test_batch=[]
            acc_test_batch=[]
            top_n_acc_valid_batch=[]
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_data = data_test[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_test[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.images: batch_data,
                                                        self.fine_labels: batch_labels,
                                                        self.bn_train: False,
                                                        self.keep_prob: 1.0,})
                loss_test_batch.append(loss)
                # y_true = np.argmax(batch_labels, axis=1)
                y_true = batch_labels
                y_pred = np.argmax(logits, axis=1)
                acc_test_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_valid_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            print('test loss: %f, test accuracy: %f, top-%d test accuracy: %f' % \
                  (np.mean(loss_test_batch), np.mean(acc_test_batch), n_top, np.mean(top_n_acc_valid_batch)))
    
    def extractor(self,
                  test_path,
                  saved_filename='feature',
                  gen_from=None, ## e.g., model_name (must given)
                  gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  out_path=None,
                  bsize=32):
        ## create output folder
        if gen_from is None:
            gen_from = os.path.join(self.result_path, self.model_name, 'models_cnn')
        if out_path is not None:
            if os.path.exists(out_path):
                print('WARNING: the output path "{}" already exists!'.format(out_path))
            else:
                os.makedirs(out_path)
        else:
            out_path = os.path.join(self.result_path, self.model_name)
        
        ## load previous model if possible
        could_load, checkpoint_counter = self.load_cnn(gen_from, gen_from_ckpt)
        if could_load:
            print(" [*] Load SUCCESS")
            
            #### load testing data
            test_dict = unpickle(test_path)
            
            ## work-around for mini-imagenet (shuffle data before extracting features)
            data_len = len(test_dict[b'fine_labels'])
            arr_all = np.arange(data_len)
            np.random.shuffle(arr_all)
            test_dict[b'data'] = test_dict[b'data'][arr_all]
            test_dict[b'fine_labels'] = [test_dict[b'fine_labels'][idx] for idx in arr_all]
            
            ### [3072] with the first 1024 being 'R', the middle 1024 being 'G', and the last 1024 being 'B'
            ### reshape and modify rank order from (batch, channel, height, width) to (batch, height, width, channel)
            data_test = test_dict[b'data'].reshape((-1, 3, self.img_size_h, self.img_size_w)).transpose([0, 2, 3, 1])
            ## work-around for mini-imagenet
            if data_test.dtype != 'uint8':
                print('Cast data from %s to uint8' % data_test.dtype)
                data_test = np.rint(data_test*255).astype('uint8')
            nBatches_test = int(np.ceil(data_test.shape[0] / bsize))
            
            ### make prediction and compute accuracy
            features_all = []
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_data = data_test[idx*bsize:(idx+1)*bsize]
                flattened = self.sess.run(self.flattened,
                                          feed_dict={self.images: batch_data,
                                                     self.bn_train: False,
                                                     self.keep_prob: 1.0,})
                features_all.append(flattened)
            features_all = np.concatenate(features_all, axis=0)
            #print('features_all.shape: %s' % (features_all.shape,))
            features_dict = {}
            features_dict[b'features'] = features_all
            if b'coarse_labels' in test_dict.keys():
                features_dict[b'coarse_labels'] = test_dict[b'coarse_labels']
            features_dict[b'fine_labels'] = test_dict[b'fine_labels']
            dopickle(features_dict, os.path.join(self.result_path, self.model_name, saved_filename))
            #return features_all ## return for debug
        
    def load(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def load_cnn(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_cnn.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
