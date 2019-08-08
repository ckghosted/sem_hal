import os, re, time, glob
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import l2_regularizer

from sklearn.metrics import accuracy_score

from ops import *
from utils import *

import pickle
import tqdm

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

# FSL that takes extracted features as inputs directly.
# (Not allow fine-tuning the CNN-based feature extractor.)
# (No need to inherit from the "VGG" class.)
# (Allow hallucination!)
class FSL_PN(object):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
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
        self.n_query = n_query
        self.n_aug = n_aug
        self.fc_dim = fc_dim
        self.n_fine_class = n_fine_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')

        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        self.hallucinated_features = self.build_hallucinator(self.triplet_features)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## The classifier is implemented as a simple 3-layer MLP with batch normalization.
    ## Just like the one used in the VGG feature extractor. But it can be re-designed.
    # def build_fsl_classifier(self, input_):
    #     with tf.variable_scope('fsl_cls', regularizer=l2_regularizer(self.l2scale)):
    #         ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
    #         self.dense14_ = self.bn_dense14_(linear(input_, self.fc_dim, name='dense14_'), train=self.bn_train) ## [-1,self.fc_dim]
    #         self.relu14_ = tf.nn.relu(self.dense14_, name='relu14_')
    #         ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
    #         self.dense15_ = self.bn_dense15_(linear(self.relu14_, self.fc_dim, name='dense15_'), train=self.bn_train) ## [-1,self.fc_dim]
    #         self.relu15_ = tf.nn.relu(self.dense15_, name='relu15_')
    #         ### Layer 16: dense with self.n_fine_class neurons, softmax
    #         self.dense16_ = linear(self.relu15_, self.n_fine_class, add_bias=True, name='dense16_') ## [-1,self.n_fine_class]
    #     return self.dense16_
    ## Simlper version of classifier: 2-layer MLP with batch normalization
    def build_fsl_classifier(self, input_):
        with tf.variable_scope('fsl_cls', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
            self.dense14_ = self.bn_dense14_(linear(input_, self.fc_dim, name='dense14_'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu14_ = tf.nn.relu(self.dense14_, name='relu14_')
            ### Layer 15: dense with self.n_fine_class neurons, softmax
            self.dense15_ = linear(self.relu14_, self.n_fine_class, add_bias=True, name='dense15_') ## [-1,self.n_fine_class]
        return self.dense15_
    
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
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim]
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
        #call: self.s_train_x_aug = self.augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        # with tf.variable_scope('hal', reuse=True, regularizer=l2_regularizer(self.l2scale)):
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
        hallucinated_features = self.build_hallucinator(triplets, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim*3)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        print('    Selected base labels: ', end='')
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            print(lb, end=', ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
                                                             np.expand_dims(features_base_train[selected_indexes[1]], 0),
                                                             np.expand_dims(seed_features[sample_count], 0)), axis=1)
        print()
        ### Forward-pass
        features_hallucinated = self.sess.run(self.hallucinated_features,
                                              feed_dict={self.triplet_features: input_features})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro(hal_from, hal_from_ckpt)

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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in range(nBatches):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train]
    
    def inference(self,
                  test_novel_path, ## test_novel_feat path (must be specified!)
                  test_base_path=None, ## test_base_feat path (if None: close-world; else: open-world)
                  gen_from=None, ## e.g., model_name (must given)
                  gen_from_ckpt=None, ## e.g., model_name+'.model-1680' (can be None)
                  out_path=None,
                  n_top=5, ## top-n accuracy
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
            
            #### load testing features
            if test_base_path:
                #### open-world
                test_novel_dict = unpickle(test_novel_path)
                test_base_dict = unpickle(test_base_path)
                features_test = np.concatenate((test_novel_dict[b'features'], test_base_dict[b'features']), axis=0)
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(test_novel_dict[b'fine_labels']).union(set(test_base_dict[b'fine_labels']))))[new_lb]] = new_lb
                fine_labels_old = [int(s) for s in test_novel_dict[b'fine_labels']+test_base_dict[b'fine_labels']]
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_old]
                fine_labels_test = np.eye(self.n_fine_class)[fine_labels_new]
                #### compute number of batches
                features_len_all = len(fine_labels_new)
                features_len_novel = len(test_novel_dict[b'fine_labels'])
                nBatches_test = int(np.ceil(features_len_all / bsize))
                #print('features_len_novel = %d' % features_len_novel)
                #print('features_len_all = %d' % features_len_all)
                #print('nBatches_test = %d' % nBatches_test)
            else:
                #### close-world
                test_dict = unpickle(test_novel_path)
                features_test = test_dict[b'features']
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_old = [int(s) for s in test_dict[b'fine_labels']]
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_old]
                fine_labels_test = np.eye(self.n_fine_class)[fine_labels_new]
                #### compute number of batches
                features_len_all = len(fine_labels_new)
                features_len_novel = features_len_all
                nBatches_test = int(np.ceil(features_len_all / bsize))
            
            ### make prediction and compute accuracy            
            loss_test_batch = []
            logits_all = None
            y_true_all = np.argmax(fine_labels_test, axis=1)
            #print('y_true_all.shape = %s' % (y_true_all.shape,))
            for idx in tqdm.tqdm(range(nBatches_test)):
                batch_features = features_test[idx*bsize:(idx+1)*bsize]
                batch_labels = fine_labels_test[idx*bsize:(idx+1)*bsize]
                loss, logits = self.sess.run([self.loss, self.logits],
                                             feed_dict={self.features: batch_features,
                                                        self.fine_labels: batch_labels,
                                                        self.bn_train: False,
                                                        self.keep_prob: 1.0,})
                loss_test_batch.append(loss)
                if logits_all is None:
                    logits_all = logits
                else:
                    logits_all = np.concatenate((logits_all, logits), axis=0)
            #print('logits_all.shape = %s' % (logits_all.shape,))
            y_pred_all = np.argmax(logits_all, axis=1)
            #print('y_pred_all.shape = %s' % (y_pred_all.shape,))
            acc_test_all = accuracy_score(y_true_all, y_pred_all)
            acc_test_novel = accuracy_score(y_true_all[0:features_len_novel], y_pred_all[0:features_len_novel])
            acc_test_base = accuracy_score(y_true_all[features_len_novel:features_len_all], y_pred_all[features_len_novel:features_len_all])
            best_n_all = np.argsort(logits_all, axis=1)[:,-n_top:]
            #print('best_n_all.shape = %s' % (best_n_all.shape,))
            top_n_acc_test_all = np.mean([(y_true_all[idx] in best_n_all[idx]) for idx in range(features_len_all)])
            top_n_acc_test_novel = np.mean([(y_true_all[idx] in best_n_all[idx]) for idx in range(features_len_novel)])
            top_n_acc_test_base = np.mean([(y_true_all[idx] in best_n_all[idx]) for idx in range(features_len_novel, features_len_all)])
            print('test loss: %f, test accuracy: %f, top-%d test accuracy: %f, novel test accuracy: %f, novel top-%d test accuracy: %f, base test accuracy: %f, base top-%d test accuracy: %f' % \
                  (np.mean(loss_test_batch), acc_test_all, n_top, top_n_acc_test_all,
                                             acc_test_novel, n_top, top_n_acc_test_novel,
                                             acc_test_base, n_top, top_n_acc_test_base))
    
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
    
    def load_hal_pro(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    
    def load_mlp(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_mlp.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

# FSL_PN with hallucinator being split into two parts:
# First, a transformation extractor that takes two base features as input and outputs a 64-dim 'transformation code';
# Second, a hallucinator that takes the novel feature and the 'transformation code' as input and outputs the hallucinated feature.
class FSL_PN_T(FSL_PN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')

        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        ### Split self.triplet_features into the first two feature vectors and the last feature vector
        self.tran_code = self.build_tran_extractor(self.triplet_features[:,0:int(self.fc_dim*2)])
        self.input_with_code = tf.concat([self.tran_code, self.triplet_features[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        self.hallucinated_features = self.build_hallucinator(self.input_with_code)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]

    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x):
        #call: self.s_train_x_aug = self.augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        # with tf.variable_scope('hal', reuse=True, regularizer=l2_regularizer(self.l2scale)):
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
        ### Split triplets into the first two feature vectors and the last feature vector
        tran_code = self.build_tran_extractor(triplets[:,0:int(self.fc_dim*2)], reuse=True)
        input_with_code = tf.concat([tran_code, triplets[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        hallucinated_features = self.build_hallucinator(input_with_code, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4
    
    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim//2, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//8, add_bias=True, name='dense2') ## [-1,self.fc_dim//8] (e.g., 64)
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
        return self.relu2

# FSL_PN_T with more complex transformation extractor
class FSL_PN_T2(FSL_PN_T):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//2, add_bias=True, name='dense2') ## [-1,self.fc_dim//2] (e.g., 256)
            self.relu2 = tf.nn.relu(self.dense2, name='relu2')
        return self.relu2

class FSL_PN_T2_VAE(FSL_PN_T2):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')

        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        ### Split self.triplet_features into the first two feature vectors and the last feature vector
        self.z_mu, self.z_logvar = self.encoder(self.triplet_features[:,0:int(self.fc_dim*2)])
        self.sampled_tran_code = self.sample_z(self.z_mu, self.z_logvar) #### shape: [-1, self.z_dim]
        self.input_with_code = tf.concat([self.sampled_tran_code, self.triplet_features[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        self.hallucinated_features = self.build_hallucinator(self.input_with_code)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug, _ = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        self.all_vars_hal_pro_enc = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name or 'enc' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro_enc = tf.train.Saver(var_list = self.all_vars_hal_pro_enc, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 512)
            relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(relu1, self.fc_dim//2, add_bias=True, name='z_mu') ## [-1,self.z_dim] (e.g., 256)
            z_logvar = linear(relu1, self.fc_dim//2, add_bias=True, name='z_logvar') ## [-1,self.z_dim] (e.g., 256)
        return z_mu, z_logvar
    
    def sample_z(self, mu, log_var, reuse=False):
        with tf.variable_scope('sample_z', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            eps = tf.random_normal(shape=tf.shape(mu))
            return mu + tf.exp(log_var / 2) * eps

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
        ### Split triplets into the first two feature vectors and the last feature vector
        z_mu, z_logvar = self.encoder(triplets[:,0:int(self.fc_dim*2)], reuse=True)
        input_z_vec = self.sample_z(z_mu, z_logvar, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim]
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
        input_with_code = tf.concat([input_z_vec, triplets[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        hallucinated_features = self.build_hallucinator(input_with_code, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, kl_loss
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro_enc(hal_from, hal_from_ckpt)

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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in range(nBatches):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train]

    def load_hal_pro_enc(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro_enc.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

class FSL_PN_T2_VAE2(FSL_PN_T2_VAE):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE2, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            dense1 = linear(input_, self.fc_dim//2, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(relu1, self.fc_dim//2, add_bias=True, name='z_mu') ## [-1,self.z_dim] (e.g., 256)
            z_logvar = linear(relu1, self.fc_dim//2, add_bias=True, name='z_logvar') ## [-1,self.z_dim] (e.g., 256)
        return z_mu, z_logvar

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
            ### Layer 3: dense with self.fc_dim neurons, ReLU
            self.dense5 = linear(self.relu4, self.fc_dim, add_bias=True, name='dense5') ## [-1,self.fc_dim]
            self.relu5 = tf.nn.relu(self.dense5, name='relu5')
        return self.relu5

class FSL_PN_T2_VAE3(FSL_PN_T2_VAE):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE3, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 512)
            relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(relu1, self.fc_dim, add_bias=True, name='z_mu') ## [-1,self.z_dim] (e.g., 512)
            z_logvar = linear(relu1, self.fc_dim, add_bias=True, name='z_logvar') ## [-1,self.z_dim] (e.g., 512)
        return z_mu, z_logvar

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
        return self.relu3

class FSL_PN_T2_VAE4(FSL_PN_T2_VAE3):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE4, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4

class FSL_PN_T2_VAE_M(FSL_PN_T2_VAE):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 n_random=5, ## number of randomly-generated transformation codes per triplet
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE_M, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
        self.n_random = n_random
        self.n_triplet = (self.n_aug - self.n_support) // self.n_random
    
    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x):
        ### make triplets
        for label_b in range(self.m_support): #### for each class in the support set
            #for n_idx in range(self.n_aug - self.n_support):
            for n_idx in range(self.n_triplet):
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
        ### Split triplets into the first two feature vectors and the last feature vector
        z_mu, z_logvar = self.encoder(triplets[:,0:int(self.fc_dim*2)], reuse=True)
        z_mu_tile = tf.reshape(tf.tile(tf.expand_dims(z_mu, 1),  [1, self.n_random, 1]), [-1, self.fc_dim//2])
        z_logvar_tile = tf.reshape(tf.tile(tf.expand_dims(z_logvar, 1),  [1, self.n_random, 1]), [-1, self.fc_dim//2])
        input_z_vec = self.sample_z(z_mu_tile, z_logvar_tile, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim]
        kl_loss = 0.5 * tf.reduce_mean(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar)
        sample_bs = triplets[:,int(self.fc_dim*2):int(self.fc_dim*3)]
        sample_bs_tile = tf.reshape(tf.tile(tf.expand_dims(sample_bs, 1),  [1, self.n_random, 1]), [-1, self.fc_dim])
        input_with_code = tf.concat([input_z_vec, sample_bs_tile], axis=1)
        hallucinated_features = self.build_hallucinator(input_with_code, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, kl_loss

class FSL_PN_T2_VAE_M2(FSL_PN_T2_VAE_M):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 n_random=5, ## number of randomly-generated transformation codes per triplet
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_VAE_M2, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    n_random,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        n_triplet_needed = (n_samples_needed // self.n_random) + 1
        print('n_samples_needed: %d, self.n_random: %d, n_triplet_needed: %d' % (n_samples_needed, self.n_random, n_triplet_needed))
        input_features = np.empty([n_triplet_needed, int(self.fc_dim*3)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        print('    Selected base labels: ', end='')
        for sample_count in range(n_triplet_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            print(lb, end=', ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
                                                             np.expand_dims(features_base_train[selected_indexes[1]], 0),
                                                             np.expand_dims(seed_features[sample_count], 0)), axis=1)
        print()
        input_features_tile = np.reshape(np.tile(np.expand_dims(input_features, 1),  [1, self.n_random, 1]), [-1, self.fc_dim*3])
        print('input_features_tile.shape: %s' % (input_features_tile.shape,))
        input_features_tile_truncated = input_features_tile[np.random.choice(int(n_triplet_needed*self.n_random), n_samples_needed, replace=False),:]
        print('input_features_tile_truncated.shape: %s' % (input_features_tile_truncated.shape,))
        ### Forward-pass
        features_hallucinated = self.sess.run(self.hallucinated_features,
                                              feed_dict={self.triplet_features: input_features_tile_truncated})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated

class FSL_PN_T2_I(FSL_PN_T2):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_I, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear_identity(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear_identity(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4

class FSL_PN_T2_IA(FSL_PN_T2_I):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_IA, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
        self.word_emb_path = word_emb_path
        self.emb_dim = emb_dim
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')

        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        self.emb_pairs = tf.placeholder(tf.float32, shape=[None]+[self.emb_dim*2], name='emb_pairs')
        ### Split self.triplet_features into the first two feature vectors and the last feature vector
        self.tran_code = self.build_tran_extractor(self.triplet_features[:,0:int(self.fc_dim*2)])
        ### Attention on the transformation code
        self.tran_code_att = self.build_semantics_attention(self.emb_pairs)
        self.tran_code_new = tf.multiply(self.tran_code, self.tran_code_att)
        self.input_with_code = tf.concat([self.tran_code_new, self.triplet_features[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        self.hallucinated_features = self.build_hallucinator(self.input_with_code)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        self.s_train_emb = tf.placeholder(tf.float32, shape=[self.m_support, self.emb_dim], name='s_train_emb')
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug = self.build_augmentor(self.s_train_x, self.s_train_emb) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        self.all_vars_hal_pro_att = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name or 'att' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro_att = tf.train.Saver(var_list = self.all_vars_hal_pro_att, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]

    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x, s_train_emb):
        #call: self.s_train_x_aug = self.augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        # with tf.variable_scope('hal', reuse=True, regularizer=l2_regularizer(self.l2scale)):
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
                emb_a = tf.squeeze(tf.slice(s_train_emb, begin=[label_a,0], size=[1,self.emb_dim])) #### shape: [self.emb_dim]
                emb_b = tf.squeeze(tf.slice(s_train_emb, begin=[label_b,0], size=[1,self.emb_dim])) #### shape: [self.emb_dim]
                emb_pair = tf.concat([emb_a, emb_b], axis=0) #### shape: [self.emb_dim*2]
                emb_pairs = tf.expand_dims(emb_pair, 0) if label_b == 0 and n_idx == 0 \
                           else tf.concat([emb_pairs, tf.expand_dims(emb_pair, 0)], axis=0) #### shape: [self.m_support*(self.n_aug-self.n_support), self.emb_dim*2]
        ### Split triplets into the first two feature vectors and the last feature vector
        tran_code = self.build_tran_extractor(triplets[:,0:int(self.fc_dim*2)], reuse=True)
        ### Attention on the transformation code
        tran_code_att = self.build_semantics_attention(emb_pairs, reuse=True)
        tran_code_new = tf.multiply(tran_code, tran_code_att)
        input_with_code = tf.concat([tran_code_new, triplets[:,int(self.fc_dim*2):int(self.fc_dim*3)]], axis=1)
        hallucinated_features = self.build_hallucinator(input_with_code, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug
    
    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//2, add_bias=True, name='dense2') ## [-1,self.fc_dim//2] (e.g., 256)
            self.relu2 = tf.nn.sigmoid(self.dense2, name='relu2')
        return self.relu2

    ## Semantics-guided attention generator
    def build_semantics_attention(self, input_, reuse=False):
        with tf.variable_scope('att', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//2, add_bias=True, name='dense2') ## [-1,self.fc_dim//2] (e.g., 256)
            self.relu2 = tf.nn.sigmoid(self.dense2, name='relu2')
        return self.relu2

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    word_emb=None,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim*3)])
        lb_emb_pair = np.empty([n_samples_needed, int(self.emb_dim*2)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        print('    Selected base labels: ', end='')
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            print(lb, end=' ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            print('(%d)' % len(candidate_indexes), end=', ')
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
                                                             np.expand_dims(features_base_train[selected_indexes[1]], 0),
                                                             np.expand_dims(seed_features[sample_count], 0)), axis=1)
            lb_emb_pair[sample_count,:] = np.concatenate((np.expand_dims(word_emb[lb[0]], 0),
                                                          np.expand_dims(word_emb[seed_fine_label], 0)), axis=1)
        print()
        ### Forward-pass
        features_hallucinated = self.sess.run(self.hallucinated_features,
                                              feed_dict={self.triplet_features: input_features, self.emb_pairs: lb_emb_pair})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated

    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        if os.path.exists(os.path.join(self.word_emb_path, 'word_emb_dict')):
            word_emb = unpickle(os.path.join(self.word_emb_path, 'word_emb_dict'))

        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro_att(hal_from, hal_from_ckpt)

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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    word_emb=word_emb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in range(nBatches):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train]
    
    def load_hal_pro_att(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro_att.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

# FSL_PN_T2 with the last activation function of the transformation extractor being sigmoid
class FSL_PN_T3(FSL_PN_T):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T3, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
        return self.relu3

    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim, add_bias=True, name='dense2') ## [-1,self.fc_dim] (e.g., 512)
            self.out = tf.nn.relu(self.dense2, name='out')
        return self.out

# FSL_PN_T2 with the last activation function of the transformation extractor being tanh
class FSL_PN_T4(FSL_PN_T):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T4, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4

    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.out = tf.nn.relu(self.dense1, name='relu1')
        return self.out

# FSL_PN_T2 with the last activation function of the transformation extractor being tanh
class FSL_PN_GAN(FSL_PN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_GAN, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
                                     l2scale)
        self.z_dim = z_dim
        self.z_std = z_std
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.features_and_noise = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim+self.z_dim], name='features_and_noise')  #### shape: [-1, self.z_dim+self.fc_dim] (e.g., 100+512=612)
        self.hallucinated_features, _ = self.build_hallucinator(self.features_and_noise)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        # self.s_train_y = tf.placeholder(tf.float32, shape=[self.m_support, 1], name='s_train_y') ### integer
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug, _ = self.build_augmentor(self.s_train_x) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
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
        hallucinated_features, hal_matrix_1 = self.build_hallucinator(input_mat, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, hal_matrix_1

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            #print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        #print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        #print('    Selected base labels: ', end='')
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            #print(lb, end=', ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            # input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
            #                                                  np.expand_dims(features_base_train[selected_indexes[1]], 0),
            #                                                  np.expand_dims(seed_features[sample_count], 0)), axis=1)
            input_features[sample_count,:] = np.expand_dims(seed_features[sample_count], 0)
        input_z = np.random.normal(loc=0.0, scale=self.z_std, size=(n_samples_needed, self.z_dim))
        features_and_noise = np.concatenate((input_z, input_features), axis=1)
        #print()
        ### Forward-pass
        features_hallucinated = self.sess.run(self.hallucinated_features,
                                              feed_dict={self.features_and_noise: features_and_noise})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated, input_z
    
    ## Add the 2nd output for the self.hallucinate() function
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              fix_seed=False,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                if fix_seed:
                    selected_indexes_per_lb = candidate_indexes_per_lb[0:n_shot]
                else:
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro(hal_from, hal_from_ckpt)

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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    used_noise_all = {}
                    features_novel_final_all = {}
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated, used_noise = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                        used_noise_all[lb] = used_noise
                        features_novel_final_all[lb] = np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in tqdm.tqdm(range(nBatches)):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train, used_noise_all, features_novel_final_all]

# Add one more layer (512x512) to the hallucinator
class FSL_PN_GAN2(FSL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_GAN2, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     z_dim,
                                     z_std,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
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

# Add one more layer (512x512) to the hallucinator
class FSL_PN_GAN_EmbMean(FSL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_GAN_EmbMean, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     z_dim,
                                     z_std,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
                                     l2scale)
        self.word_emb_path = word_emb_path
        self.emb_dim = emb_dim

    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        # self.features_and_noise = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim+self.z_dim], name='features_and_noise')  #### shape: [-1, self.z_dim+self.fc_dim] (e.g., 100+512=612)
        self.input_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.emb = tf.placeholder(tf.float32, shape=[None]+[self.emb_dim], name='emb')
        self.noise = self.sample_z(self.emb, self.z_std) #### shape: [-1, self.z_dim]
        self.features_and_noise = tf.concat([self.noise, self.input_features], axis=1)
        self.hallucinated_features, _ = self.build_hallucinator(self.features_and_noise)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        self.s_train_emb = tf.placeholder(tf.float32, shape=[self.m_support, self.emb_dim], name='s_train_emb')
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug, _ = self.build_augmentor(self.s_train_x, self.s_train_emb) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro = tf.train.Saver(var_list = self.all_vars_hal_pro, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]

    def sample_z(self, mu, z_std, reuse=False):
        with tf.variable_scope('sample_z', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            eps = tf.random_normal(shape=tf.shape(mu), stddev=z_std)
            return mu + eps

    ## "For each class, we use G to generate n_gen additional examples till there are exactly n_aug examples per class." (Y-X Wang, 2018)
    def build_augmentor(self, s_train_x, s_train_emb):
        ### make input matrix
        for label_b in range(self.m_support): #### for each class in the support set
            for n_idx in range(self.n_aug - self.n_support):
                #### (1) Randomly select one sample as seed
                sample_b = tf.slice(s_train_x, begin=[label_b,np.random.choice(self.n_support, 1)[0],0], size=[1,1,self.fc_dim]) #### shape: [1, 1, self.fc_dim]
                sample_b = tf.squeeze(sample_b, [0,1]) #### shape: [self.fc_dim]
                #### (2) Append a noise vector
                input_z_vec = self.sample_z(s_train_emb[label_b], self.z_std, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim]
                input_vec = tf.concat([input_z_vec, sample_b], axis=0) #### shape: [self.z_dim+self.fc_dim] (e.g., 100+512=612)
                #### (3) Make input matrix
                input_mat = tf.expand_dims(input_vec, 0) if label_b == 0 and n_idx == 0 \
                            else tf.concat([input_mat, tf.expand_dims(input_vec, 0)], axis=0) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim+self.fc_dim]
        hallucinated_features, hal_matrix_1 = self.build_hallucinator(input_mat, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, hal_matrix_1

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    word_emb=None,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            #print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        #print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        #print('    Selected base labels: ', end='')
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            #print(lb, end=', ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            # input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
            #                                                  np.expand_dims(features_base_train[selected_indexes[1]], 0),
            #                                                  np.expand_dims(seed_features[sample_count], 0)), axis=1)
            input_features[sample_count,:] = np.expand_dims(seed_features[sample_count], 0)
        emb_tile = np.tile(np.expand_dims(word_emb[seed_fine_label], 0), [n_samples_needed,1])
        #print('seed_fine_label: %d' % seed_fine_label)
        #print('word_emb[seed_fine_label]: ', end='')
        #print(word_emb[seed_fine_label])
        #print('emb_tile has shape: %s' % (emb_tile.shape,))
        #print()
        ### Forward-pass
        features_hallucinated, input_z = self.sess.run([self.hallucinated_features, self.noise],
                                              feed_dict={self.input_features: input_features, self.emb: emb_tile})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated, input_z

    ## Add the 2nd output for the self.hallucinate() function
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        if os.path.exists(os.path.join(self.word_emb_path, 'word_emb_dict')):
            word_emb = unpickle(os.path.join(self.word_emb_path, 'word_emb_dict'))

        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)
        
        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro(hal_from, hal_from_ckpt)

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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    used_noise_all = {}
                    features_novel_final_all = {}
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated, used_noise = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    word_emb=word_emb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                        used_noise_all[lb] = used_noise
                        features_novel_final_all[lb] = np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in range(nBatches):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train, used_noise_all, features_novel_final_all]

# FSL_PN_T2 with the last activation function of the transformation extractor being tanh
class FSL_PN_VAEGAN(FSL_PN_GAN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_VAEGAN, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     z_dim,
                                     z_std,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
                                     l2scale)
        self.word_emb_path = word_emb_path
        self.emb_dim = emb_dim
    
    def build_model(self):
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        self.learning_rate_aux = tf.placeholder(tf.float32, shape=[], name='learning_rate_aux')
        
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        self.bn_pro = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_pro')
        
        print("build model started")
        ### (1) The main task: 100-class classification
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        # self.logits = self.build_fsl_classifier(self.features)
        self.features_encode = self.build_proto_encoder(self.features)
        self.logits = self.build_fsl_classifier(self.features_encode)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        # self.features_and_noise = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim+self.z_dim], name='features_and_noise')  #### shape: [-1, self.z_dim+self.fc_dim] (e.g., 100+512=612)
        self.input_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.emb = tf.placeholder(tf.float32, shape=[None]+[self.emb_dim], name='emb')
        self.z_mu, self.z_logvar = self.encoder(self.emb)
        self.noise = self.sample_z(self.z_mu, self.z_logvar) #### shape: [-1, self.z_dim]
        self.features_and_noise = tf.concat([self.noise, self.input_features], axis=1)
        self.hallucinated_features, _ = self.build_hallucinator(self.features_and_noise)
        ### (2) The auxiliary task: m-way classification
        self.s_train_x = tf.placeholder(tf.float32, shape=[self.m_support, self.n_support, self.fc_dim], name='s_train_x')
        self.s_train_emb = tf.placeholder(tf.float32, shape=[self.m_support, self.emb_dim], name='s_train_emb')
        self.s_test_x = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='s_test_x')
        self.s_test_y = tf.placeholder(tf.int32, shape=[None], name='s_test_y') ### integer (which class of the support set does the test sample belong to?)
        self.s_test_y_vec = tf.one_hot(self.s_test_y, self.m_support)
        self.s_train_x_aug, _ = self.build_augmentor(self.s_train_x, self.s_train_emb) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_x_aug_reshape = tf.reshape(self.s_train_x_aug, shape=[-1, self.fc_dim]) ### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.proto_enc_in = tf.concat([self.s_train_x_aug_reshape, self.s_test_x], axis=0) #### shape: [self.m_support*self.n_aug+self.n_query, self.fc_dim]
        self.proto_enc_out = self.build_proto_encoder(self.proto_enc_in, reuse=True)
        self.s_train_x_aug_encode = tf.slice(self.proto_enc_out, begin=[0, 0], size=[self.m_support*self.n_aug, self.fc_dim]) #### shape: [self.m_support*self.n_aug, self.fc_dim]
        self.s_train_x_aug_encode = tf.reshape(self.s_train_x_aug_encode, shape=[self.m_support, self.n_aug, self.fc_dim]) ### shape: [self.m_support, self.n_aug, self.fc_dim]
        self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug_encode, axis=1) ### shape: [self.m_support, self.fc_dim]
        self.s_test_x_encode = tf.slice(self.proto_enc_out, begin=[self.m_support*self.n_aug, 0], size=[self.n_query, self.fc_dim]) #### shape: [self.n_query, self.fc_dim]
        self.s_test_x_tile = tf.reshape(tf.tile(self.s_test_x_encode, multiples=[1, self.m_support]), [ self.n_query, self.m_support, self.fc_dim]) #### shape: [self.n_query, self.m_support, self.fc_dim]
        # self.s_train_prototypes = tf.reduce_mean(self.s_train_x_aug, axis=1) ### shape: [self.m_support, self.fc_dim]
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        ### (1) The main task: 100-class classification
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        ### (2) The auxiliary task: m-way classification
        self.logits_aux = -tf.norm(self.s_train_prototypes - self.s_test_x_tile, ord='euclidean', axis=2) ### shape: [self.n_query, self.m_support]
        self.loss_aux = tf.nn.softmax_cross_entropy_with_logits(labels=self.s_test_y_vec,
                                                                logits=self.logits_aux,
                                                                name='loss_aux')
        self.acc_aux = tf.nn.in_top_k(self.logits_aux, self.s_test_y, k=1)
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.all_vars_hal_pro = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name)]
        self.all_vars_hal_pro_enc = [var for var in self.all_vars if ('hal' in var.name or 'pro' in var.name or 'enc' in var.name)]
        ### trainable variables
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        self.trainable_vars_hal_pro = [var for var in self.trainable_vars if ('hal' in var.name or 'pro' in var.name)]
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        self.used_regs_hal_pro = [reg for reg in self.all_regs if \
                              ('hal' in reg.name or 'pro' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        self.opt_hal_pro = tf.train.AdamOptimizer(learning_rate=self.learning_rate_aux,
                                                  beta1=0.5).minimize(self.loss_aux+sum(self.used_regs_hal_pro),
                                                                      var_list=self.trainable_vars_hal_pro)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal_pro_enc = tf.train.Saver(var_list = self.all_vars_hal_pro_enc, max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
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
        z_mu, z_logvar = self.encoder(s_train_emb_tile, reuse=True)
        input_z_vec = self.sample_z(z_mu, z_logvar, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.z_dim]
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
        hallucinated_features, hal_matrix_1 = self.build_hallucinator(input_mat, reuse=True) #### shape: [self.m_support*(self.n_aug-self.n_support), self.fc_dim]
        hallucinated_features = tf.reshape(hallucinated_features, shape=[self.m_support, -1, self.fc_dim]) #### shape: [self.m_support, self.n_aug-self.n_support, self.fc_dim]
        s_train_x_aug = tf.concat([s_train_x, hallucinated_features], axis=1) #### shape: [self.m_support, self.n_aug, self.fc_dim]
        return s_train_x_aug, hal_matrix_1

    def hallucinate(self,
                    seed_features,
                    seed_coarse_lb,
                    n_samples_needed,
                    train_base_path,
                    coarse_specific,
                    hal_from,
                    seed_fine_label,
                    word_emb=None,
                    similar_lb_dict=None):
        #print(" [***] Hallucinator Load SUCCESS")
        ### Load training features and labels of the base classes
        ### (Take the first 80% since the rest are used for validation in the train() function)
        #### [20181025] Just take all 100% since we are not running validation during FSL training
        train_base_dict = unpickle(train_base_path)
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]

        ### Use base classes belonging to the same coarse class (specified by 'seed_coarse_lb') only
        if seed_coarse_lb is not None:
            #print('--- [novel class %d belongs to superclass %d]' % (seed_fine_label, seed_coarse_lb))
            coarse_base_train = [int(s) for s in train_base_dict[b'coarse_labels']]
            same_coarse_indexes = [idx for idx in range(len(coarse_base_train)) \
                                   if coarse_base_train[idx] == seed_coarse_lb]
            features_base_train = features_base_train[same_coarse_indexes]
            labels_base_train = [labels_base_train[idx] for idx in same_coarse_indexes]
            #### Further, we may want to have coarse-specific hallucinators
            #### [20181025] Not used anymore since worse performance and less general
            if coarse_specific:
                print('Load hallucinator for coarse label %02d...' % seed_coarse_lb)
                hal_from_basename = os.path.basename(hal_from)
                hal_from_folder = os.path.abspath(os.path.join(hal_from, '..'))
                hal_from_new = os.path.join(hal_from_folder + '_coarse%02d' % seed_coarse_lb, hal_from_basename)
                could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from_new, None)
        
        ### Create a batch of size "n_samples_needed", with each row being consisted of
        ### (base_feature1, base_feature2, seed_feature), where base_feature1 and base_feature2
        ### are randomly selected from the same base class.
        input_features = np.empty([n_samples_needed, int(self.fc_dim)])
        all_possible_base_lbs = list(set(labels_base_train))
        if similar_lb_dict:
            #### [spacy] Select a base class from the set of "similar" classes specified by "similar_lb_dict", if given
            all_possible_base_lbs = similar_lb_dict[seed_fine_label]
        #print('Hallucinating novel class %d using base classes %s' % (seed_fine_label, all_possible_base_lbs))
        #print('    Selected base labels: ', end='')
        for sample_count in range(n_samples_needed):
            #### (1) Randomly select a base class
            lb = np.random.choice(all_possible_base_lbs, 1)
            #print(lb, end=', ')
            #### (2) Randomly select two samples from the above base class
            candidate_indexes = [idx for idx in range(len(labels_base_train)) if labels_base_train[idx] == lb]
            selected_indexes = np.random.choice(candidate_indexes, 2, replace=False) #### Use replace=False to avoid two identical base samples
            #### (3) Concatenate (base_feature1, base_feature2, seed_feature) to form a row of the model input
            ####     Note that seed_feature has shape (1, fc_dim) already ==> no need np.expand_dims()
            # input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
            #                                                  np.expand_dims(features_base_train[selected_indexes[1]], 0),
            #                                                  np.expand_dims(seed_features[sample_count], 0)), axis=1)
            input_features[sample_count,:] = np.expand_dims(seed_features[sample_count], 0)
        emb_tile = np.tile(np.expand_dims(word_emb[seed_fine_label], 0), [n_samples_needed,1])
        #print('seed_fine_label: %d' % seed_fine_label)
        #print('word_emb[seed_fine_label]: ', end='')
        #print(word_emb[seed_fine_label])
        #print('emb_tile has shape: %s' % (emb_tile.shape,))
        #print()
        ### Forward-pass
        features_hallucinated, input_z = self.sess.run([self.hallucinated_features, self.noise],
                                              feed_dict={self.input_features: input_features, self.emb: emb_tile})
        ### Choose the hallucinated features with high probability of the correct fine_label
        #self.logits_temp = self.build_mlp(self.features_temp)
        #logits_hallucinated = self.sess.run(self.logits_temp,
        #                                    feed_dict={self.features_temp: features_hallucinated})
        #print('logits_hallucinated.shape: %s' % (logits_hallucinated.shape,))
        return features_hallucinated, input_z
    
    def train(self,
              train_novel_path, ## train_novel_feat path (must be specified!)
              train_base_path, ## train_base_feat path (must be specified!)
              hal_from, ## e.g., hal_name (must given)
              #mlp_from, ## e.g., mlp_name (must given)
              hal_from_ckpt=None, ## e.g., hal_name+'.model-1680' (can be None)
              #mlp_from_ckpt=None, ## e.g., mlp_name+'.model-1680' (can be None)
              data_path=None, ## Path of the saved class_mapping or similar_labels dictionaries (if None, don't consider coarse labels or semantics-based label dependencies for hallucination)
              sim_set_path=None, ## Path of the saved sim_set_sorted dictionary (if None, don't consider semantics-based label dependencies for hallucination)
              sim_thre_frac=5,
              min_base_labels=1,
              max_base_labels=5,
              coarse_specific=False, ## if True, use coarse-label-specific hallucinators
              n_shot=1,
              n_min=20, ## minimum number of samples per training class ==> (n_min - n_shot) more samples need to be hallucinated
              n_top=5, ## top-n accuracy
              bsize=32,
              learning_rate=3e-6,
              learning_rate_aux=3e-7,
              num_epoch=10,
              num_epoch_per_hal=2,
              n_iteration_aux=20,
              fix_seed=False,
              train_hal_from_scratch=False):
        ### create a dedicated folder for this model
        if os.path.exists(os.path.join(self.result_path, self.model_name)):
            print('WARNING: the folder "{}" already exists!'.format(os.path.join(self.result_path, self.model_name)))
        else:
            os.makedirs(os.path.join(self.result_path, self.model_name))
            os.makedirs(os.path.join(self.result_path, self.model_name, 'models'))
        
        ### Load training features (as two dictionaries) from both base and novel classes
        train_novel_dict = unpickle(train_novel_path)
        features_novel_train = train_novel_dict[b'features']
        labels_novel_train = [int(s) for s in train_novel_dict[b'fine_labels']]
        # labels_novel_train = np.eye(self.n_fine_class)[fine_labels]
        train_base_dict = unpickle(train_base_path)
        features_len_per_base = int(len(train_base_dict[b'fine_labels']) / len(set(train_base_dict[b'fine_labels'])))
        features_base_train = train_base_dict[b'features']
        labels_base_train = [int(s) for s in train_base_dict[b'fine_labels']]
        # labels_base_train = np.eye(self.n_fine_class)[fine_labels]
        
        similar_lb_dict = None
        class_mapping_inv = None
        ### Load similar_labels dictionary if possible
        if os.path.exists(os.path.join(sim_set_path, 'sim_set_sorted')):
            ## Load the sorted similarities between all pairs of label-name vectors, each in the following form:
            ## (label_index_1, label_index_2, label_name_1, label_name_2, similarity), e.g.,
            ## (59, 52, 'pine_tree', 'oak_tree', 0.9333858489990234)
            sim_set_sorted = unpickle(os.path.join(sim_set_path, 'sim_set_sorted'))
            ## Make set of similar labels for each label
            ## [Note] Must consider base labels
            all_novel_labels = set(labels_novel_train)
            all_base_labels = set(labels_base_train)
            similar_lb_dict = {}
            similar_counter = 0
            lb_not_enough_sim = []
            threshold = sim_thre_frac / 10.0
            print('threshold = %f' % threshold)
            for fine_lb_idx in all_novel_labels: ### Different from make_quaddruplet_similar.py, we need to find similar labels for all novel labels
                ### For each label, collect all its similarity results (note: w.r.t. base labels)
                sim_set_for_this = [item for item in sim_set_sorted if item [0] == fine_lb_idx and item[1] in all_base_labels]
                ### Consider similar labels with similarity > threshold
                sim_lb_candidate = [item[1] for item in sim_set_for_this if item [4] > threshold]
                if len(sim_lb_candidate) > max_base_labels:
                    #### If there are too many similar labels, only take the first ones
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate[0:max_base_labels]
                elif len(sim_lb_candidate) < min_base_labels:
                    #### If there are not enough similar labels, take the ones with the most similarity values
                    #### by re-defining candidate similar labels
                    sim_lb_candidate_more = [item[1] for item in sim_set_for_this]
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate_more[0:min_base_labels]
                    lb_not_enough_sim.append(fine_lb_idx)
                else:
                    similar_lb_dict[fine_lb_idx] = sim_lb_candidate
                print('%d: ' % fine_lb_idx, end='')
                print(similar_lb_dict[fine_lb_idx])
                similar_counter = similar_counter + len(similar_lb_dict[fine_lb_idx])
            print('similar_counter = %d' % similar_counter)
        ### Otherwise, load the {Superclass: {Classes}} dictionary if possible
        elif os.path.exists(os.path.join(data_path, 'class_mapping')):
            class_mapping = unpickle(os.path.join(data_path, 'class_mapping'))
            #### Make an inverse mapping from (novel) fine labels to the corresponding coarse labels
            class_mapping_inv = {}
            for fine_lb in set(train_novel_dict[b'fine_labels']):
                for coarse_lb in class_mapping.keys():
                    if fine_lb in class_mapping[coarse_lb]:
                        class_mapping_inv[fine_lb] = coarse_lb
                        break
            #print('class_mapping_inv:')
            #print(class_mapping_inv)
        
        if n_shot >= n_min:
            #### Hallucination not needed
            selected_indexes = []
            for lb in set(labels_novel_train):
                ##### Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes.extend(selected_indexes_per_lb)
            features_novel_final = features_novel_train[selected_indexes]
            labels_novel_final = [labels_novel_train[idx] for idx in selected_indexes]
        else:
            #### Hallucination needed
            selected_indexes_novel = {}
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                if fix_seed:
                    selected_indexes_per_lb = candidate_indexes_per_lb[0:n_shot]
                else:
                    selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_indexes_novel[lb] = selected_indexes_per_lb
        
        if os.path.exists(os.path.join(self.word_emb_path, 'word_emb_dict')):
            word_emb = unpickle(os.path.join(self.word_emb_path, 'word_emb_dict'))

        ## initialization
        initOp = tf.global_variables_initializer()
        self.sess.run(initOp)

        ## load previous trained hallucinator and mlp linear classifier
        if (not coarse_specific) and (not train_hal_from_scratch):
            could_load_hal_pro, checkpoint_counter_hal = self.load_hal_pro_enc(hal_from, hal_from_ckpt)
        
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
            ### ==================================================================================================
            ### Re-hallucinate and fine-tune hallucinator by the auxiliary task for every num_epoch_per_hal epochs
            ### ==================================================================================================
            # if train_hal_from_scratch or (num_epoch_per_hal == 1) or (num_epoch_per_hal == 0 and epoch == 1) or (num_epoch_per_hal > 0 and epoch%num_epoch_per_hal == 1):
            if epoch == 1:
                ### For the training split, use all base samples and randomly selected novel samples.
                if n_shot < n_min:
                    #### Hallucination needed
                    n_features_novel_final = int(n_min * len(set(labels_novel_train)))
                    features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
                    used_noise_all = {}
                    features_novel_final_all = {}
                    # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
                    labels_novel_final = []
                    lb_counter = 0
                    for lb in set(labels_novel_train):
                        ##### (1) Randomly select n-shot features from each class
                        selected_indexes_per_lb = selected_indexes_novel[lb]
                        selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                        ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                        seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                        seed_features = features_novel_train[seed_indexes]
                        seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                        ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                        if (not train_hal_from_scratch) and (not coarse_specific) and (not could_load_hal_pro):
                            print('Load hallucinator or mlp linear classifier fail!!!!!!')
                            feature_hallucinated = seed_features
                        else:
                            feature_hallucinated, used_noise = self.hallucinate(seed_features=seed_features,
                                                                    seed_coarse_lb=seed_coarse_lb,
                                                                    n_samples_needed=n_min-n_shot,
                                                                    train_base_path=train_base_path,
                                                                    coarse_specific=coarse_specific,
                                                                    hal_from=hal_from,
                                                                    seed_fine_label=lb,
                                                                    word_emb=word_emb,
                                                                    similar_lb_dict=similar_lb_dict)
                            print('feature_hallucinated.shape: %s' % (feature_hallucinated.shape,))
                        features_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = \
                            np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                        # labels_novel_final[lb_counter*n_min:(lb_counter+1)*n_min,:] = np.eye(self.n_fine_class)[np.repeat(lb, n_min)]
                        labels_novel_final.extend([lb for _ in range(n_min)])
                        lb_counter += 1
                        used_noise_all[lb] = used_noise
                        features_novel_final_all[lb] = np.concatenate((selected_features_per_lb, feature_hallucinated), axis=0)
                ### Before concatenating the (repeated or hallucinated) novel dataset and the base dataset,
                ### repeat the novel dataset to balance novel/base
                #print('features_len_per_base = %d' % features_len_per_base)
                features_novel_balanced = np.repeat(features_novel_final, int(features_len_per_base/n_min), axis=0)
                # labels_novel_balanced = np.repeat(labels_novel_final, int(features_len_per_base/n_min), axis=0)
                labels_novel_balanced = []
                for lb in labels_novel_final:
                    labels_novel_balanced.extend([lb for _ in range(int(features_len_per_base/n_min))])
                
                ### [20181025] We are not running validation during FSL training since it is meaningless
                features_train = np.concatenate((features_novel_balanced, features_base_train), axis=0)
                # fine_labels_train = np.concatenate((labels_novel_balanced, labels_base_train), axis=0)
                fine_labels_train = labels_novel_balanced + labels_base_train
                nBatches = int(np.ceil(features_train.shape[0] / bsize))
                print('features_train.shape: %s' % (features_train.shape,))
                ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
                ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
                label_mapping = {}
                for new_lb in range(self.n_fine_class):
                    label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
                fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
                fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]
                ### Features indexes used to shuffle training order
                arr = np.arange(features_train.shape[0])

            ### shuffle training order for each epoch
            np.random.shuffle(arr)
            #print('training')
            for idx in tqdm.tqdm(range(nBatches)):
                batch_features = features_train[arr[idx*bsize:(idx+1)*bsize]]
                batch_labels = fine_labels_train[arr[idx*bsize:(idx+1)*bsize]]
                #print(batch_labels.shape)
                _, loss, logits = self.sess.run([self.opt_fsl_cls, self.loss, self.logits],
                                                feed_dict={self.features: batch_features,
                                                           self.fine_labels: batch_labels,
                                                           self.bn_train: True,
                                                           self.keep_prob: 0.5,
                                                           self.learning_rate: learning_rate})
                loss_train_batch.append(loss)
                y_true = np.argmax(batch_labels, axis=1)
                y_pred = np.argmax(logits, axis=1)
                acc_train_batch.append(accuracy_score(y_true, y_pred))
                best_n = np.argsort(logits, axis=1)[:,-n_top:]
                top_n_acc_train_batch.append(np.mean([(y_true[batch_idx] in best_n[batch_idx]) for batch_idx in range(len(y_true))]))
            ### [20181025] We are not running validation during FSL training since it is meaningless
            ### record training loss for each epoch (instead of each iteration)
            loss_train.append(np.mean(loss_train_batch))
            acc_train.append(np.mean(acc_train_batch))
            top_n_acc_train.append(np.mean(top_n_acc_train_batch))
            print('Epoch: %d, train loss: %f, train accuracy: %f, top-%d train accuracy: %f' % \
                  (epoch, np.mean(loss_train_batch), np.mean(acc_train_batch), n_top, np.mean(top_n_acc_train_batch)))
        
        ## [20181025] We are not running validation during FSL training since it is meaningless.
        ## Just save the final model
        self.saver.save(self.sess,
                        os.path.join(self.result_path, self.model_name, 'models', self.model_name + '.model'),
                        global_step=epoch)
        
        return [loss_train, acc_train, used_noise_all, features_novel_final_all]

    def load_hal_pro_enc(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal_pro_enc.restore(self.sess, os.path.join(init_from, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

# HAL_PN_VAEGAN with a simpler encoder (directly mapping 300-dim embedding vector of label into 512-dim mean and logvar)
class FSL_PN_VAEGAN2(FSL_PN_VAEGAN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_VAEGAN2, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     word_emb_path,
                                     emb_dim,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     z_dim,
                                     z_std,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
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

# HAL_PN_VAEGAN with a simpler encoder (directly mapping 300-dim embedding vector of label into 512-dim mean and logvar)
class FSL_PN_VAEGAN2_NoBias(FSL_PN_VAEGAN):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 word_emb_path=None,
                 emb_dim=300,
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 z_dim=100,
                 z_std=1.0,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_VAEGAN2_NoBias, self).__init__(sess,
                                     model_name,
                                     result_path,
                                     word_emb_path,
                                     emb_dim,
                                     m_support,
                                     n_support,
                                     n_aug,
                                     n_query,
                                     fc_dim,
                                     z_dim,
                                     z_std,
                                     n_fine_class,
                                     bnDecay,
                                     epsilon,
                                     l2scale)
    
    def encoder(self, input_, reuse=False):
        with tf.variable_scope('enc', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            # dense1 = linear(input_, self.z_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            # relu1 = tf.nn.relu(dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            z_mu = linear(input_, self.z_dim, add_bias=False, name='z_mu') ## [-1,self.z_dim] (e.g., 100)
            z_logvar = linear(input_, self.z_dim, add_bias=False, name='z_logvar') ## [-1,self.z_dim] (e.g., 100)
        return z_mu, z_logvar

# FSL_PN_T with more complex transformation extractor
class FSL_PN_Q(FSL_PN_T):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_Q, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense3 = linear(input_, self.fc_dim, add_bias=True, name='dense3') ## [-1,self.fc_dim]
            self.relu3 = tf.nn.relu(self.dense3, name='relu3')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense4 = linear(self.relu3, self.fc_dim, add_bias=True, name='dense4') ## [-1,self.fc_dim]
            self.relu4 = tf.nn.relu(self.dense4, name='relu4')
        return self.relu4

    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim//2, add_bias=True, name='dense1') ## [-1,self.fc_dim//2] (e.g., 256)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//8, add_bias=True, name='dense2') ## [-1,self.fc_dim//8] (e.g., 64)
            self.prob2 = tf.nn.sigmoid(self.dense2, name='prob2')
        return self.prob2

# FSL_PN_T with more complex transformation extractor
class FSL_PN_T2_Q(FSL_PN_T2):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
                 m_support=40, ## number of classes in the support set
                 n_support=5, ## number of samples per class in the support set
                 n_aug=20, ## number of samples per class in the augmented support set
                 n_query=200, ## number of samples in the query set
                 fc_dim=512,
                 n_fine_class=50,
                 bnDecay=0.9,
                 epsilon=1e-5,
                 l2scale=0.001):
        super(FSL_PN_T2_Q, self).__init__(sess,
                                    model_name,
                                    result_path,
                                    m_support,
                                    n_support,
                                    n_aug,
                                    n_query,
                                    fc_dim,
                                    n_fine_class,
                                    bnDecay,
                                    epsilon,
                                    l2scale)
    
    ## Transformation extractor
    def build_tran_extractor(self, input_, reuse=False):
        with tf.variable_scope('hal', reuse=reuse, regularizer=l2_regularizer(self.l2scale)):
            ### Layer 1: dense with self.fc_dim neurons, ReLU
            self.dense1 = linear(input_, self.fc_dim, add_bias=True, name='dense1') ## [-1,self.fc_dim] (e.g., 512)
            self.relu1 = tf.nn.relu(self.dense1, name='relu1')
            ### Layer 2: dense with self.fc_dim neurons, ReLU
            self.dense2 = linear(self.relu1, self.fc_dim//2, add_bias=True, name='dense2') ## [-1,self.fc_dim//2] (e.g., 256)
            self.prob2 = tf.nn.sigmoid(self.dense2, name='relu2')
        return self.prob2