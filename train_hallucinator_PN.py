import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_HAL_PN import HAL_PN, HAL_PN_GAN, HAL_PN_GAN2, HAL_PN_VAEGAN, HAL_PN_VAEGAN2
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

import pickle

def dopickle(dict_, file):
    with open(file, 'wb') as fo:
        pickle.dump(dict_, fo)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_emb_path', default='/data/put_data/cclin/datasets/cifar-100-python', type=str, help='Path of the saved semantic information')
    parser.add_argument('--emb_dim', default=300, type=int, help='Dimensions of the saved semantic information')
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--data_path', type=str, help='Path of the saved class_mapping dictionary')
    parser.add_argument('--extractor_name', type=str, help='Folder name of the saved extractor model')
    parser.add_argument('--hallucinator_name', type=str, help='Folder name to save hallucinator models and learning curves')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--m_support', default=40, type=int, help='Number of classes in the support set')
    parser.add_argument('--n_support', default=5, type=int, help='Number of samples per class in the support set')
    parser.add_argument('--n_aug', default=20, type=int, help='Number of samples per class in the augmented support set')
    parser.add_argument('--n_query', default=100, type=int, help='Number of samples in the query set')
    parser.add_argument('--num_epoch', default=500, type=int, help='Max number of training epochs')
    parser.add_argument('--n_ite_per_epoch', default=100, type=int, help='Number of iterations (episodes) per epoch')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
    parser.add_argument('--lambda_kl', default=0.001, type=float, help='lambda_kl')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early-stopping mechanism')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--VAEGAN', action='store_true', help='Use VAEGAN-based hallucinator if present')
    parser.add_argument('--VAEGAN2', action='store_true', help='Use VAEGAN2-based hallucinator if present')
    parser.add_argument('--use_coarse', action='store_true', help='Use HAL_coarse (hallucinator with a transformation extractor) if present')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    args = parser.parse_args()
    train(args)

# Train the hallucinator
def train(args):
    train_base_path = os.path.join(args.result_path, args.extractor_name, 'train_base_feat')
        
    tf.reset_default_graph()
    with tf.Session() as sess:
        if args.GAN:
            # HAL_PN_GAN: implementation of (Y.-X. Wang, CVPR 2018)
            net = HAL_PN_GAN(sess,
                      model_name=args.hallucinator_name,
                      result_path=args.result_path,
                      m_support=args.m_support,
                      n_support=args.n_support,
                      n_aug=args.n_aug,
                      n_query=args.n_query,
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      l2scale=args.l2scale)
        elif args.GAN2:
            # HAL_PN_GAN2: adds one more layer to the hallucinator of HAL_PN_GAN
            net = HAL_PN_GAN2(sess,
                      model_name=args.hallucinator_name,
                      result_path=args.result_path,
                      m_support=args.m_support,
                      n_support=args.n_support,
                      n_aug=args.n_aug,
                      n_query=args.n_query,
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      l2scale=args.l2scale)
        elif args.VAEGAN:
            # HAL_PN_VAEGAN: implementation of our idea (C.-C. Lin, ICIP 2019)
            net = HAL_PN_VAEGAN(sess,
                      model_name=args.hallucinator_name,
                      result_path=args.result_path,
                      word_emb_path=args.word_emb_path,
                      emb_dim=args.emb_dim,
                      m_support=args.m_support,
                      n_support=args.n_support,
                      n_aug=args.n_aug,
                      n_query=args.n_query,
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      lambda_kl=args.lambda_kl,
                      l2scale=args.l2scale)
        elif args.VAEGAN2:
            # HAL_PN_VAEGAN2: simpler version of encoder than HAL_PN_VAEGAN
            net = HAL_PN_VAEGAN2(sess,
                      model_name=args.hallucinator_name,
                      result_path=args.result_path,
                      word_emb_path=args.word_emb_path,
                      emb_dim=args.emb_dim,
                      m_support=args.m_support,
                      n_support=args.n_support,
                      n_aug=args.n_aug,
                      n_query=args.n_query,
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      lambda_kl=args.lambda_kl,
                      l2scale=args.l2scale)
        elif args.use_coarse:
            net = HAL_PN_COARSE(sess,
                      model_name=args.hallucinator_name,
                      result_path=args.result_path,
                      m_support=args.m_support,
                      n_support=args.n_support,
                      n_aug=args.n_aug,
                      n_query=args.n_query,
                      fc_dim=args.fc_dim,
                      l2scale=args.l2scale)
        else:
            # HAL_PN: combine the analogy-based hallucinator (Hariharan, ICCV 2017) with the meta-learning-based hallucinator (Y.-X. Wang, CVPR 2018)
            net = HAL_PN(sess,
                  model_name=args.hallucinator_name,
                  result_path=args.result_path,
                  m_support=args.m_support,
                  n_support=args.n_support,
                  n_aug=args.n_aug,
                  n_query=args.n_query,
                  fc_dim=args.fc_dim,
                  l2scale=args.l2scale)
        all_vars, trainable_vars, all_regs = net.build_model()
        # Debug: Check trainable variables and regularizers
        if args.debug:
            print('------------------[all_vars]------------------')
            for var in all_vars:
                print(var.name)
            print('------------------[trainable_vars]------------------')
            for var in trainable_vars:
                print(var.name)
            print('------------------[all_regs]------------------')
            for var in all_regs:
                print(var.name)
        if args.use_coarse:
            res = net.train(train_base_path=train_base_path,
                            data_path=args.data_path,
                            learning_rate=args.learning_rate,
                            num_epoch=args.num_epoch,
                            n_ite_per_epoch=args.n_ite_per_epoch,
                            patience=args.patience)
        else:
            res = net.train(train_base_path=train_base_path,
                            learning_rate=args.learning_rate,
                            num_epoch=args.num_epoch,
                            n_ite_per_epoch=args.n_ite_per_epoch,
                            patience=args.patience)
    np.save(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'), res)
    
    # Plot learning curve
    results = np.load(os.path.join(args.result_path, args.hallucinator_name, 'results.npy'))
    fig, ax = plt.subplots(1,2, figsize=(15,6))
    ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
    ax[0].plot(range(1, len(results[1])+1), results[1], label='Validation error')
    ax[0].set_xticks(np.arange(1, len(results[0])+1))
    ax[0].set_xlabel('Training epochs', fontsize=16)
    ax[0].set_ylabel('Cross entropy', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[1].plot(range(1, len(results[2])+1), results[2], label='Training accuracy')
    ax[1].plot(range(1, len(results[3])+1), results[3], label='Validation accuracy')
    ax[1].set_xticks(np.arange(1, len(results[2])+1))
    ax[1].set_xlabel('Training epochs', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].legend(fontsize=16)
    plt.suptitle('Learning Curve', fontsize=20)
    fig.savefig(os.path.join(args.result_path, args.hallucinator_name, 'learning_curve.jpg'),
                bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    main()
