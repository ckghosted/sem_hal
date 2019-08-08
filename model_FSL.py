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
class FSL(object):
    def __init__(self,
                 sess,
                 model_name='FSL',
                 result_path='/home/cclin/few_shot_learning/hallucination_by_analogy/results',
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
        self.fc_dim = fc_dim
        self.n_fine_class = n_fine_class
        self.bnDecay = bnDecay
        self.epsilon = epsilon
        self.l2scale = l2scale
    
    def build_model(self):
        self.features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features')
        self.coarse_labels = tf.placeholder(tf.float32, shape=[None], name='coarse_labels')
        self.fine_labels = tf.placeholder(tf.float32, shape=[None]+[self.n_fine_class], name='fine_labels')
        self.bn_train = tf.placeholder('bool', name='bn_train')
        self.keep_prob = tf.placeholder(tf.float32, shape=[], name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        
        ### Used the classifier on the base classes learnt during representation learning
        ### to compute the probability of the correct fine_label of hallucinated features.
        #self.bn_dense14 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14')
        #self.bn_dense15 = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15')
        ### The classifier is implemented as a simple 3-layer MLP with batch normalization.
        ### Just like the one used in the VGG feature extractor. But it can be re-designed.
        self.bn_dense14_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense14_')
        self.bn_dense15_ = batch_norm(epsilon=self.epsilon, momentum=self.bnDecay, name='bn_dense15_')
        print("build model started")
        self.logits = self.build_fsl_classifier(self.features)
        ### Also build the mlp.
        ### No need to define loss or optimizer since we only need foward-pass
        #self.features_temp = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim], name='features_temp')
        #self.logits_temp = self.build_mlp(self.features_temp)
        ### Also build the hallucinator.
        ### No need to define loss or optimizer since we only need foward-pass
        self.triplet_features = tf.placeholder(tf.float32, shape=[None]+[self.fc_dim*3], name='triplet_features')
        self.hallucinated_features = self.build_hallucinator(self.triplet_features)
        print("build model finished, define loss and optimizer")
        
        ### Compute accuracy (optional)
        #self.outputs = tf.nn.softmax(self.dense16) ## [-1,self.n_fine_class]
        #self.pred = tf.argmax(self.outputs, axis=1) ## [-1,1]
        #self.acc = tf.reduce_mean(tf.cast(tf.equal(self.pred, self.fine_labels), tf.float32))
        
        ### Define loss and training ops
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.fine_labels,
                                                                           logits=self.logits,
                                                                           name='loss'))
        
        ### variables
        self.all_vars = tf.global_variables()
        self.all_vars_hal = [var for var in self.all_vars if 'hal' in var.name]
        #self.all_vars_mlp = [var for var in self.all_vars if 'mlp' in var.name]
        self.all_vars_fsl_cls = [var for var in self.all_vars if 'fsl_cls' in var.name]
        self.trainable_vars = tf.trainable_variables()
        self.trainable_vars_fsl_cls = [var for var in self.trainable_vars if 'fsl_cls' in var.name]
        
        ### regularizers
        self.all_regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.used_regs = [reg for reg in self.all_regs if \
                          ('filter' in reg.name) or ('Matrix' in reg.name) or ('bias' in reg.name)]
        self.used_regs_fsl_cls = [reg for reg in self.all_regs if \
                                  ('fsl_cls' in reg.name) and (('Matrix' in reg.name) or ('bias' in reg.name))]
        
        ### optimizers
        self.opt_fsl_cls = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                  beta1=0.5).minimize(self.loss+sum(self.used_regs_fsl_cls),
                                                                      var_list=self.trainable_vars_fsl_cls)
        #self.opt_all = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
        #                                      beta1=0.5).minimize(self.loss+sum(self.used_regs),
        #                                                          var_list=self.trainable_vars)
        
        ### Create model saver (keep the best 3 checkpoint)
        self.saver = tf.train.Saver(max_to_keep = 1)
        self.saver_hal = tf.train.Saver(var_list = self.all_vars_hal,
                                        max_to_keep = 1)
        #self.saver_mlp = tf.train.Saver(var_list = self.all_vars_mlp,
        #                                max_to_keep = 1)
        return [self.all_vars, self.trainable_vars, self.all_regs]
    
    ## Used the classifier on the base classes learnt during representation learning
    ## to compute the probability of the correct fine_label of hallucinated features.
    #def build_mlp(self, input_):
    #    with tf.variable_scope('mlp', regularizer=l2_regularizer(self.l2scale)):
    #        ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
    #        self.dense14 = self.bn_dense14(linear(input_, self.fc_dim, name='dense14'), train=self.bn_train) ## [-1,self.fc_dim]
    #        self.relu14 = tf.nn.relu(self.dense14, name='relu14')
    #        ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
    #        self.dense15 = self.bn_dense15(linear(self.relu14, self.fc_dim, name='dense15'), train=self.bn_train) ## [-1,self.fc_dim]
    #        self.relu15 = tf.nn.relu(self.dense15, name='relu15')
    #        ### Layer 16: dense with self.n_fine_class neurons, softmax
    #        self.dense16 = linear(self.relu15, self.n_fine_class, add_bias=True, name='dense16') ## [-1,self.n_fine_class]
    #    return self.dense16
    
    ## The classifier is implemented as a simple 3-layer MLP with batch normalization.
    ## Just like the one used in the VGG feature extractor. But it can be re-designed.
    def build_fsl_classifier(self, input_):
        with tf.variable_scope('fsl_cls', regularizer=l2_regularizer(self.l2scale)):
            ### Layer 14: dense with self.fc_dim neurons, BN, and ReLU
            self.dense14_ = self.bn_dense14_(linear(input_, self.fc_dim, name='dense14_'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu14_ = tf.nn.relu(self.dense14_, name='relu14_')
            ### Layer 15: dense with self.fc_dim neurons, BN, and ReLU
            self.dense15_ = self.bn_dense15_(linear(self.relu14_, self.fc_dim, name='dense15_'), train=self.bn_train) ## [-1,self.fc_dim]
            self.relu15_ = tf.nn.relu(self.dense15_, name='relu15_')
            ### Layer 16: dense with self.n_fine_class neurons, softmax
            self.dense16_ = linear(self.relu15_, self.n_fine_class, add_bias=True, name='dense16_') ## [-1,self.n_fine_class]
        return self.dense16_

    ## "For our generator G, we use a three layer MLP with ReLU as the activation function" (Hariharan, 2017)
    def build_hallucinator(self, input_):
        with tf.variable_scope('hal', regularizer=l2_regularizer(self.l2scale)):
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
        input_features = np.empty([n_samples_needed, int(self.fc_dim*3)])
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
            input_features[sample_count,:] = np.concatenate((np.expand_dims(features_base_train[selected_indexes[0]], 0),
                                                             np.expand_dims(features_base_train[selected_indexes[1]], 0),
                                                             np.expand_dims(seed_features[sample_count], 0)), axis=1)
        #print()
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
              learning_rate=5e-5,
              num_epoch=50,
              patience=10):
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
        
        ## load previous trained hallucinator and mlp linear classifier
        if not coarse_specific:
            could_load_hal, checkpoint_counter_hal = self.load_hal(hal_from, hal_from_ckpt)
        #could_load_mlp, checkpoint_counter_mlp = self.load_mlp(mlp_from, mlp_from_ckpt)

        ### For the training split, use all base samples and randomly selected novel samples.
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
            n_features_novel_final = int(n_min * len(set(labels_novel_train)))
            features_novel_final = np.empty([n_features_novel_final, self.fc_dim])
            # labels_novel_final = np.empty([n_features_novel_final, self.n_fine_class])
            labels_novel_final = []
            lb_counter = 0
            for lb in set(labels_novel_train):
                ##### (1) Randomly select n-shot features from each class
                candidate_indexes_per_lb = [idx for idx in range(len(labels_novel_train)) \
                                            if labels_novel_train[idx] == lb]
                selected_indexes_per_lb = np.random.choice(candidate_indexes_per_lb, n_shot)
                selected_features_per_lb = features_novel_train[selected_indexes_per_lb]
                ##### (2) Randomly select n_min-n_shot seed features (from the above n-shot samples) for hallucination
                seed_indexes = np.random.choice(selected_indexes_per_lb, n_min-n_shot, replace=True)
                seed_features = features_novel_train[seed_indexes]
                seed_coarse_lb = class_mapping_inv[lb] if class_mapping_inv else None
                ##### (3) Collect (n_shot) selected features and (n_min - n_shot) hallucinated features
                if (not coarse_specific) and (not could_load_hal):
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
        
        ### Features indexes used to shuffle training order
        arr = np.arange(features_train.shape[0])
        
        ### Before create one-hot vectors for labels, make a dictionary for {old_label: new_label} mapping,
        ### e.g., {0:0, 3:1, 5:2, 6:3, 7:4, ..., 99:49}, such that all labels become 0~49
        label_mapping = {}
        for new_lb in range(self.n_fine_class):
            label_mapping[np.sort(list(set(fine_labels_train)))[new_lb]] = new_lb
        fine_labels_new = [label_mapping[old_lb] for old_lb in fine_labels_train]
        fine_labels_train = np.eye(self.n_fine_class)[fine_labels_new]

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
    
    def load_hal(self, init_from, init_from_ckpt=None):
        ckpt = tf.train.get_checkpoint_state(init_from)
        if ckpt and ckpt.model_checkpoint_path:
            if init_from_ckpt is None:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            else:
                ckpt_name = init_from_ckpt
            self.saver_hal.restore(self.sess, os.path.join(init_from, ckpt_name))
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
