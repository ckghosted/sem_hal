'''
[cclin 2019/08/02]
1. Only HAL_PN, HAL_PN_GAN, HAL_PN_GAN2, HAL_PN_VAEGAN, HAL_PN_VAEGAN2 are finally used.
   (HAL_PN_VAEGAN2_NoBias make no significant difference to HAL_PN_VAEGAN2)
2. HAL_PN is originally designed to combine the analogy-based hallucinator (Hariharan, ICCV 2017) with
   the meta-learning-based hallucinator (Y.-X. Wang, CVPR 2018). As we can see, the 'build_augmentor()'
   member function makes 'triplets' as the input to the 'build_hallucinator()' member function.
3. HAL_PN_GAN is the implementation of (Y.-X. Wang, CVPR 2018).
4. HAL_PN_GAN2 adds one more layer to the hallucinator of HAL_PN_GAN.
5. HAL_PN_VAEGAN is the implementation of our idea (C.-C. Lin, ICIP 2019).
6. HAL_PN_VAEGAN2 uses a simpler version of encoder than HAL_PN_VAEGAN.
'''
import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer
import tqdm

from ops import *
from utils import *

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# Use base classes to train the hallucinator
class HAL_PN(object):
    def __init__(self,
                 sess,
                 model_name='HAL_P',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=5, ## number of classes in the support set
                 n_support=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query=3, ## number of samples in the query set
                 fc_dim=512,
                 # n_fine_class=80,
                 # loss_lambda=10,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        self.sess = sess
        self.model_name = model_name
        self.result_path = result_path
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)
        self.m_support = m_support
        self.n_support = n_support
        self.n_aug = n_aug
        self.n_query = n_query
        self.fc_dim = fc_dim
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
    
    def build_model(self):
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        self.s_train_x_aug = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Define loss and training ops
        self.logits = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits,
                                                                name='loss')
        self.acc = tf.nn.in_top_k(self.logits, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        
        ### Create model saver (keep the best 3 checkpoint)
        # self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## "For PN, the embedding architecture consists of two MLP layers with ReLU as the activation function." (Y-X Wang, 2018)
    def build_proto_encoder(self, input_, reuse=False):
        with tf.variable_scope('pro', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1_, self.pro_matrix_1, self.pro_bias_1 = linear(input_, self.fc_dim, add_bias=True, name='dense1', with_w=True) ## [-1,self.fc_dim]
            self.dense1 = self.bn_pro(self.dense1_, train=self.bn_train)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
        return self.relu2
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1, self.hal_matrix_1, self.hal_bias_1 = linear(input_, self.fc_dim, add_bias=True, name='dense1', with_w=True) ## [-1,self.fc_dim]
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
            ### Layer 3: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(self.relu2, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
        return self.relu3
    
    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x):
        ### make triplets
        for label_b in range(self.m_support): #### for each class in the support set
            for n_idx in range(self.n_aug - self.n_support):
                #### (1) Randomly select one sample as seed
                sample_b = tf.slice(s_train_x, begin=[label_b,np.random.choice(self.n_support, 1)[0],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_b = tf.squeeze(sample_b, [0,1]) #### shape: [self.fc_dim]
                #### (2) Randomly select another class, from which randomly select two samples
                label_a = np.random.choice(list(set(np.arange(self.m_support)) - set([label_b])), 1)[0]
                sample_a_idx = np.random.choice(self.n_support, 2, replace=False)
                sample_a1 = tf.slice(s_train_x, begin=[label_a,sample_a_idx[0],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_a1 = tf.squeeze(sample_a1, [0,1]) #### shape: [self.fc_dim]
                sample_a2 = tf.slice(s_train_x, begin=[label_a,sample_a_idx[1],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_a2 = tf.squeeze(sample_a2, [0,1]) #### shape: [self.fc_dim]
                #### (3) Make a triplet
                triplet = tf.concat([sample_a1, sample_a2, sample_b], axis=0) #### shape: [self.fc_dim*3]
                triplets = tf.expand_dims(triplet, 0) if label_b == 0 and n_idx == 0 \
                           else tf.concat([triplets, tf.expand_dims(triplet, 0)], axis=0) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim*3]
        hallucinated_features = self.build_hallucinator(triplets) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug

    def train(self,
              train_base_path,
              learning_rate=5e-5,
              num_epoch=5,
              n_ite_per_epoch=2,
              patience=3):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            # os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models_hal_pro'))
        
        ### Load all base-class data as training data,
        ### and split each of them into training/validation by 80/20
        train_base_dict = unpickle(train_base_path)
        data_len = len(train_base_dict[b'fine_labels'])
        features_base_train = train_base_dict[b'features'][0:int(data_len*0.8)]
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels'][0:int(data_len*0.8)]]
        features_base_valid = train_base_dict[b'features'][int(data_len*0.8):data_len]
        labels_base_valid = [int(s) for s in train_base_dict[b'fine_labels'][int(data_len*0.8):data_len]]
        print('set(labels_base_train): %s' % (set(labels_base_train),))
        print('set(labels_base_valid): %s' % (set(labels_base_valid),))
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            ### Training
            all_base_labels = set(labels_base_train)
            loss_ite_train = []
            loss_ite_valid = []
            acc_ite_train = []
            acc_ite_valid = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_train)) \
                                                if labels_base_train[idx] == lb]
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_train[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_train[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                _, loss, logits, s_train_x_aug, acc = self.sess.run([self.opt_hal_pro, self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: True,
                                                                   self.learning_rate: learning_rate})
                loss_ite_train.append(loss)
                acc_ite_train.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            ### Validation
            all_base_labels = set(labels_base_valid)
            for ite in tqdm.tqdm(range(1, int(n_ite_per_epoch*2+1))):
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_valid)) \
                                                if labels_base_valid[idx] == lb]
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_valid[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_valid[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                loss, logits, s_train_x_aug, acc = self.sess.run([self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: False})
                loss_ite_valid.append(loss)
                acc_ite_valid.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            loss_train.append(np.mean(loss_ite_train))
            loss_valid.append(np.mean(loss_ite_valid))
            acc_train.append(np.mean(acc_ite_train))
            acc_valid.append(np.mean(acc_ite_valid))
            print('---- Epoch: %d, training loss: %f, training accuracy: %f, validation loss: %f, validation accuracy: %f' % \
                (epoch, np.mean(loss_ite_train), np.mean(acc_ite_train), np.mean(loss_ite_valid), np.mean(acc_ite_valid)))
            
            #### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_ite_valid)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    # self.saver.save(self.sess,
                                    # os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    # global_step=epoch)
                    self.saver_hal_pro.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]

# Use base classes to train the hallucinator
class HAL_PN_GAN(HAL_PN):
    def __init__(self,
                 sess,
                 model_name='HAL_P',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=5, ## number of classes in the support set
                 n_support=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query=3, ## number of samples in the query set
                 fc_dim=512,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 z_dim=100,
                 z_std=1.0,
                 l2scale=0.001):
        super(HAL_PN_GAN, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         m_support,
                                         n_support,
                                         n_aug,
                                         n_query,
                                         fc_dim,
                                         bnDecay,
                                         epsilon,
                                         l2scale)
        self.z_dim = z_dim
        self.z_std = z_std
    
    def build_model(self):
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        self.s_train_x_aug, self.hal_matrix_1 = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Define loss and training ops
        self.logits = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits,
                                                                name='loss')
        self.acc = tf.nn.in_top_k(self.logits, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        
        ### Create model saver (keep the best 3 checkpoint)
        # self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1, self.hal_matrix_1, self.hal_bias_1 = linear_identity(input_, self.fc_dim, add_bias=True, name='dense1', with_w=True) ## [-1,self.fc_dim]
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear_identity(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
            ### Layer 3: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear_identity(self.relu2, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
        return self.relu3, self.hal_matrix_1
    
    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x):
        ### make input matrix
        for label_b in range(self.m_support): #### for each class in the support set
            for n_idx in range(self.n_aug - self.n_support):
                #### (1) Randomly select one sample as seed
                sample_b = tf.slice(s_train_x, begin=[label_b,np.random.choice(self.n_support, 1)[0],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_b = tf.squeeze(sample_b, [0,1]) #### shape: [self.fc_dim]
                #### (2) Append a noise vector
                input_z = tf.random_normal([self.z_dim], stddev=self.z_std)
                input_vec = tf.concat([input_z, sample_b], axis=0) #### shape: [self.z_dim+self.fc_dim] (e.g., 100+512=612)
                #### (3) Make input matrix
                input_mat = tf.expand_dims(input_vec, 0) if label_b == 0 and n_idx == 0 \
                            else tf.concat([input_mat, tf.expand_dims(input_vec, 0)], axis=0) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim+self.fc_dim]
        hallucinated_features, hal_matrix_1 = self.build_hallucinator(input_mat) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, hal_matrix_1

    def train(self,
              train_base_path,
              learning_rate=5e-5,
              num_epoch=5,
              n_ite_per_epoch=2,
              patience=3):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            # os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models_hal_pro'))
        
        ### Load all base-class data as training data,
        ### and split each of them into training/validation by 80/20
        train_base_dict = unpickle(train_base_path)
        data_len = len(train_base_dict[b'fine_labels'])
        ### 2019/05/10: shuffle before spliting!
        arr = np.arange(data_len)
        np.random.shuffle(arr)
        features_base_train = train_base_dict[b'features'][arr[0:int(data_len*0.8)]]
        labels_base_train = [int(train_base_dict[b'fine_labels'][idx]) for idx in arr[0:int(data_len*0.8)]]
        features_base_valid = train_base_dict[b'features'][arr[int(data_len*0.8):data_len]]
        labels_base_valid = [int(train_base_dict[b'fine_labels'][idx]) for idx in arr[int(data_len*0.8):data_len]]
        print('set(labels_base_train): %s' % (set(labels_base_train),))
        print('set(labels_base_valid): %s' % (set(labels_base_valid),))
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            ### Training
            all_base_labels = set(labels_base_train)
            loss_ite_train = []
            loss_ite_valid = []
            acc_ite_train = []
            acc_ite_valid = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                #### [Debug] Inspect the weight matrix in the first hidden layer of hallucinator
                # if ite == 1:
                #     hal_matrix_1 = self.sess.run(self.hal_matrix_1)
                #     print('hal_matrix_1.shape = %s' % (hal_matrix_1.shape,))
                #     print(hal_matrix_1)
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                #### Workaround for AwA (or other dataset with imbalanced distribution among categories): Specify the probability for each label being sampled
                # selected_lbs_prob = [len([idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]) / len(labels_base_train) for lb in selected_lbs]
                # selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True, p=selected_lbs_prob)
                #### 2019/05/10: don't specify p for imagenet1000
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                skip_this_episode = False
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_train)) \
                                                if labels_base_train[idx] == lb]
                    if len(candidate_indexes_per_lb) < self.n_support+n_test_samples_per_lb:
                        print('[Training] Skip this episode since there are not enough samples for label %d, which has only %d samples but %d are needed' % (lb, len(candidate_indexes_per_lb), self.n_support+n_test_samples_per_lb))
                        skip_this_episode = True
                        break
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_train[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_train[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                if skip_this_episode:
                    continue
                _, loss, logits, s_train_x_aug, acc = self.sess.run([self.opt_hal_pro, self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: True,
                                                                   self.learning_rate: learning_rate})
                loss_ite_train.append(loss)
                acc_ite_train.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            ### Validation
            all_base_labels = set(labels_base_valid)
            for ite in tqdm.tqdm(range(1, int(n_ite_per_epoch*2+1))):
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                #### Workaround for AwA (or other dataset with imbalanced distribution among categories): Specify the probability for each label being sampled
                # selected_lbs_prob = [len([idx for idx in range(len(labels_base_valid)) if labels_base_valid[idx] == lb]) / len(labels_base_valid) for lb in selected_lbs]
                # selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True, p=selected_lbs_prob)
                #### 2019/05/10: don't specify p for imagenet1000
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                skip_this_episode = False
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_valid)) \
                                                if labels_base_valid[idx] == lb]
                    if len(candidate_indexes_per_lb) < self.n_support+n_test_samples_per_lb:
                        print('[Validation] Skip this episode since there are not enough samples for label %d, which has only %d samples but %d are needed' % (lb, len(candidate_indexes_per_lb), self.n_support+n_test_samples_per_lb))
                        skip_this_episode = True
                        break
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_valid[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_valid[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                if skip_this_episode:
                    continue
                loss, logits, s_train_x_aug, acc = self.sess.run([self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: False})
                loss_ite_valid.append(loss)
                acc_ite_valid.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            loss_train.append(np.mean(loss_ite_train))
            loss_valid.append(np.mean(loss_ite_valid))
            acc_train.append(np.mean(acc_ite_train))
            acc_valid.append(np.mean(acc_ite_valid))
            print('---- Epoch: %d, training loss: %f, training accuracy: %f, validation loss: %f, validation accuracy: %f' % \
                (epoch, np.mean(loss_ite_train), np.mean(acc_ite_train), np.mean(loss_ite_valid), np.mean(acc_ite_valid)))
            
            #### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_ite_valid)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    # self.saver.save(self.sess,
                                    # os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    # global_step=epoch)
                    self.saver_hal_pro.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_hal_pro', self.model_name + '.model-hal-pro'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]

# Add one more layer (512x512) to the hallucinator
class HAL_PN_GAN2(HAL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name='HAL_P',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=5, ## number of classes in the support set
                 n_support=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query=3, ## number of samples in the query set
                 fc_dim=512,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 z_dim=100,
                 z_std=1.0,
                 l2scale=0.001):
        super(HAL_PN_GAN2, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         m_support,
                                         n_support,
                                         n_aug,
                                         n_query,
                                         fc_dim,
                                         bnDecay,
                                         epsilon,
                                         z_dim,
                                         z_std,
                                         l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1, self.hal_matrix_1, self.hal_bias_1 = linear_identity(input_, self.fc_dim, add_bias=True, name='dense1', with_w=True) ## [-1,self.fc_dim]
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear_identity(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim]
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
            ### Layer 3: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear_identity(self.relu2, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 4: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear_identity(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4, self.hal_matrix_1

# Use base classes to train the hallucinator
class HAL_PN_VAEGAN(HAL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name='HAL_P',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 lambda_kl=0.001,
                 m_support=5, ## number of classes in the support set
                 n_support=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query=3, ## number of samples in the query set
                 fc_dim=512,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 z_dim=100,
                 z_std=1.0,
                 l2scale=0.001):
        super(HAL_PN_VAEGAN, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         m_support,
                                         n_support,
                                         n_aug,
                                         n_query,
                                         fc_dim,
                                         bnDecay,
                                         epsilon,
                                         z_dim,
                                         z_std,
                                         l2scale)
        self.word_emb_path = word_emb_path
        self.emb_dim = emb_dim
        self.lambda_kl = lambda_kl
    
    def build_model(self):
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        self.s_train_emb = tf.placeholder(tf.float32, shape=[self.m_support, self.emb_dim], name='s_train_emb')
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        self.s_train_x_aug, self.hal_matrix_1, self.kl_loss = self.build_augmentor(self.s_train_x, self.s_train_emb) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Define loss and training ops
        self.logits = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits,
                                                                name='loss')
        self.acc = tf.nn.in_top_k(self.logits, self.s_test_y, k=1)
        
        ### Add KL-divergence loss
        self.loss_all = self.loss + self.lambda_kl * self.kl_loss

        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal_pro_enc = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name or 'enc' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_hal_pro_enc = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name or 'enc' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_hal_pro_enc = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name or 'enc' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_hal_pro_enc = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss_all+sum(self.used_regs_hal_pro_enc),
                                                                      var_list=self.trainable_vars_hal_pro_enc)
        
        ### Create model saver (keep the best 3 checkpoint)
        # self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro_enc = tf.train.Saver(var_list = self.all_vars_hal_pro_enc, max_to_keep = 1)
        
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            dense1 = linear(input_, self.fc_dim//2, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(relu1, self.z_dim, add_bias=True, name='z_mu') ## [-1,self.z_dim] (e.g., 100)
            z_logvar = linear(relu1, self.z_dim, add_bias=True, name='z_logvar') ## [-1,self.z_dim] (e.g., 100)
        return z_mu, z_logvar
    
    def sample_z(self, mu, log_var, reuse=False):
        with tf.variable_scope('sample_z', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(log_var / 2) * eps

    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x, s_train_emb):
        #### VAE-GAN: sample random gaussian parametrized by the encoder and the word embedding of the label name
        s_train_emb_tile = tf.reshape(tf.tile(tf.expand_dims(s_train_emb, 1),  [1, self.n_aug - self.n_support, 1]), [-1, self.emb_dim]) #### shape: [self.m_support*(self.n_aug-self.n_support), self.emb_dim]
        z_mu, z_logvar = self.encoder(s_train_emb_tile)
        input_z_vec = self.sample_z(z_mu, z_logvar) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim]
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
        ### make input matrix
        for label_b in range(self.m_support): #### for each class in the support set
            for n_idx in range(self.n_aug - self.n_support):
                #### (1) Randomly select one sample as seed
                sample_b = tf.slice(s_train_x, begin=[label_b,np.random.choice(self.n_support, 1)[0],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_b = tf.squeeze(sample_b, [0,1]) #### shape: [self.fc_dim]
                #### (2) Make sample matrix
                sample_bs = tf.expand_dims(sample_b, 0) if label_b == 0 and n_idx == 0 \
                            else tf.concat([sample_bs, tf.expand_dims(sample_b, 0)], axis=0) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        ### Make input matrix
        input_mat = tf.concat([input_z_vec, sample_bs], axis=1) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim+self.fc_dim]
        hallucinated_features, hal_matrix_1 = self.build_hallucinator(input_mat) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, hal_matrix_1, kl_loss

    def train(self,
              train_base_path,
              learning_rate=5e-5,
              num_epoch=5,
              n_ite_per_epoch=2,
              patience=3):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            # os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models_hal_pro_enc'))
        
        ### Load all base-class data as training data,
        ### and split each of them into training/validation by 80/20
        train_base_dict = unpickle(train_base_path)
        data_len = len(train_base_dict[b'fine_labels'])
        ### 2019/05/10: shuffle before spliting!
        arr = np.arange(data_len)
        np.random.shuffle(arr)
        features_base_train = train_base_dict[b'features'][arr[0:int(data_len*0.8)]]
        labels_base_train = [int(train_base_dict[b'fine_labels'][idx]) for idx in arr[0:int(data_len*0.8)]]
        features_base_valid = train_base_dict[b'features'][arr[int(data_len*0.8):data_len]]
        labels_base_valid = [int(train_base_dict[b'fine_labels'][idx]) for idx in arr[int(data_len*0.8):data_len]]
        print('set(labels_base_train): %s' % (set(labels_base_train),))
        print('set(labels_base_valid): %s' % (set(labels_base_valid),))

        if os.path.exists(os.path.join(self.word_emb_path, 'word_emb_dict')):
            word_emb = unpickle(os.path.join(self.word_emb_path, 'word_emb_dict'))
        
        ### initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ### main training loop
        loss_train = []
        loss_valid = []
        acc_train = []
        acc_valid = []
        best_loss = 0
        stopping_step = 0
        for epoch in range(1, (num_epoch+1)):
            ### Training
            all_base_labels = set(labels_base_train)
            loss_ite_train = []
            loss_ite_valid = []
            acc_ite_train = []
            acc_ite_valid = []
            for ite in tqdm.tqdm(range(1, (n_ite_per_epoch+1))):
                #### [Debug] Inspect the weight matrix in the first hidden layer of hallucinator
                # if ite == 1:
                #     hal_matrix_1 = self.sess.run(self.hal_matrix_1)
                #     print('hal_matrix_1.shape = %s' % (hal_matrix_1.shape,))
                #     print(hal_matrix_1)
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_train_emb = np.empty([self.m_support, self.emb_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                #### Workaround for AwA (or other dataset with imbalanced distribution among categories): Specify the probability for each label being sampled
                # selected_lbs_prob = [len([idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]) / len(labels_base_train) for lb in selected_lbs]
                # selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True, p=selected_lbs_prob)
                #### 2019/05/10: don't specify p for imagenet1000
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                skip_this_episode = False
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_train)) \
                                                if labels_base_train[idx] == lb]
                    if len(candidate_indexes_per_lb) < self.n_support+n_test_samples_per_lb:
                        print('[Training] Skip this episode since there are not enough samples for label %d, which has only %d samples but %d are needed' % (lb, len(candidate_indexes_per_lb), self.n_support+n_test_samples_per_lb))
                        skip_this_episode = True
                        break
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_train[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_train[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                    s_train_emb[lb_idx,:] = word_emb[lb]
                if skip_this_episode:
                    continue
                _, loss, logits, s_train_x_aug, acc = self.sess.run([self.opt_hal_pro_enc, self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_train_emb: s_train_emb,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: True,
                                                                   self.learning_rate: learning_rate})
                loss_ite_train.append(loss)
                acc_ite_train.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            ### Validation
            all_base_labels = set(labels_base_valid)
            for ite in tqdm.tqdm(range(1, int(n_ite_per_epoch*2+1))):
                s_train_x = np.empty([self.m_support, self.n_support, self.fc_dim])
                s_train_emb = np.empty([self.m_support, self.emb_dim])
                s_test_x = np.empty([self.n_query, self.fc_dim])
                s_test_y = np.empty([self.n_query], dtype=int)
                #### (1) Sample self.m_support classes from the set of base classes, and at most self.n_support examples per class
                selected_lbs = np.random.choice(list(all_base_labels), self.m_support, replace=False)
                #### Workaround for AwA (or other dataset with imbalanced distribution among categories): Specify the probability for each label being sampled
                # selected_lbs_prob = [len([idx for idx in range(len(labels_base_valid)) if labels_base_valid[idx] == lb]) / len(labels_base_valid) for lb in selected_lbs]
                # selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True, p=selected_lbs_prob)
                #### 2019/05/10: don't specify p for imagenet1000
                selected_lbs_for_test = np.random.choice(selected_lbs, self.n_query, replace=True)
                test_lb_counter = 0
                skip_this_episode = False
                for lb_idx in range(self.m_support):
                    lb = selected_lbs[lb_idx]
                    n_test_samples_per_lb = np.sum([lb_test == lb for lb_test in selected_lbs_for_test])
                    # print('---- for lb %d, n_test_samples_per_lb = %d' % (lb, n_test_samples_per_lb))
                    candidate_indexes_per_lb = [idx for idx in range(len(labels_base_valid)) \
                                                if labels_base_valid[idx] == lb]
                    if len(candidate_indexes_per_lb) < self.n_support+n_test_samples_per_lb:
                        print('[Validation] Skip this episode since there are not enough samples for label %d, which has only %d samples but %d are needed' % (lb, len(candidate_indexes_per_lb), self.n_support+n_test_samples_per_lb))
                        skip_this_episode = True
                        break
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, self.n_support+n_test_samples_per_lb, replace=False)
                    s_train_x[lb_idx,:,:] = features_base_valid[selected_indexes_per_lb[0:self.n_support]]
                    s_test_x[test_lb_counter:(test_lb_counter+n_test_samples_per_lb),:] = \
                        features_base_valid[selected_indexes_per_lb[self.n_support:self.n_support+n_test_samples_per_lb]]
                    s_test_y[test_lb_counter:(test_lb_counter+n_test_samples_per_lb)] = np.repeat(lb_idx, n_test_samples_per_lb)
                    # print('np.repeat(lb_idx, n_test_samples_per_lb): %s' % np.repeat(lb_idx, n_test_samples_per_lb))
                    test_lb_counter = test_lb_counter + n_test_samples_per_lb
                    s_train_emb[lb_idx,:] = word_emb[lb]
                if skip_this_episode:
                    continue
                loss, logits, s_train_x_aug, acc = self.sess.run([self.loss, self.logits, self.s_train_x_aug, self.acc],
                                                        feed_dict={self.s_train_x: s_train_x,
                                                                   self.s_train_emb: s_train_emb,
                                                                   self.s_test_x: s_test_x,
                                                                   self.s_test_y: s_test_y,
                                                                   self.bn_train: False})
                loss_ite_valid.append(loss)
                acc_ite_valid.append(acc)
                # print('Ite: %d, support set: %s, test sample: %s (y=%s), s_train_x_aug.shape = %s\nmodel output logit: %s, acc_aux = %s' % \
                   # (ite, selected_lbs, selected_lbs_for_test, s_test_y, s_train_x_aug.shape, logits_aux, acc_aux))
            loss_train.append(np.mean(loss_ite_train))
            loss_valid.append(np.mean(loss_ite_valid))
            acc_train.append(np.mean(acc_ite_train))
            acc_valid.append(np.mean(acc_ite_valid))
            print('---- Epoch: %d, training loss: %f, training accuracy: %f, validation loss: %f, validation accuracy: %f' % \
                (epoch, np.mean(loss_ite_train), np.mean(acc_ite_train), np.mean(loss_ite_valid), np.mean(acc_ite_valid)))
            
            #### save model if improvement, stop if reach patience
            current_loss = np.mean(loss_ite_valid)
            if epoch == 1:
                best_loss = current_loss
            else:
                if current_loss < best_loss: ## only monitor loss
                    best_loss = current_loss
                    # self.saver.save(self.sess,
                                    # os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                                    # global_step=epoch)
                    self.saver_hal_pro_enc.save(self.sess,
                                        os.path.join(self.result_path, self.model_name, 'models_hal_pro_enc', self.model_name + '.model-hal-pro-enc'),
                                        global_step=epoch)
                    stopping_step = 0
                else:
                    stopping_step += 1
                print('stopping_step = %d' % stopping_step)
                if stopping_step >= patience:
                    print('stopping_step >= patience (%d), stop training' % patience)
                    break
        return [loss_train, loss_valid, acc_train, acc_valid]

# HAL_PN_VAEGAN with a simpler encoder (directly mapping 300-dim embedding vector of label into 512-dim mean and logvar)
class HAL_PN_VAEGAN2(HAL_PN_VAEGAN):
    def __init__(self,
                 sess,
                 model_name='HAL_P',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 lambda_kl=0.001,
                 m_support=5, ## number of classes in the support set
                 n_support=2, ## number of samples per class in the support set
                 n_aug=4, ## number of samples per class in the augmented support set
                 n_query=3, ## number of samples in the query set
                 fc_dim=512,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 z_dim=100,
                 z_std=1.0,
                 l2scale=0.001):
        super(HAL_PN_VAEGAN2, self).__init__(sess,
                                         model_name,
                                         result_path,
                                         word_emb_path,
                                         emb_dim,
                                         lambda_kl,
                                         m_support,
                                         n_support,
                                         n_aug,
                                         n_query,
                                         fc_dim,
                                         bnDecay,
                                         epsilon,
                                         z_dim,
                                         z_std,
                                         l2scale)
    
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            # dense1 = linear(input_, self.z_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            # relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(input_, self.z_dim, add_bias=True, name='z_mu') ## [-1,self.z_dim] (e.g., 100)
            z_logvar = linear(input_, self.z_dim, add_bias=True, name='z_logvar') ## [-1,self.z_dim] (e.g., 100)
        return z_mu, z_logvar
