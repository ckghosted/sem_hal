import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_FSL_PN import FSL_PN, FSL_PN_GAN, FSL_PN_GAN2, FSL_PN_VAEGAN, FSL_PN_VAEGAN2
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_emb_path', default='/data/put_data/cclin/datasets/cifar-100-python', type=str, help='Path of the saved semantic information')
    parser.add_argument('--emb_dim', default=300, type=int, help='Dimensions of the saved semantic information')
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--model_name', type=str, help='Folder name to save FSL models and learning curves')
    parser.add_argument('--extractor_name', type=str, help='Folder name of the saved extractor model')
    parser.add_argument('--hallucinator_name', default='Baseline', type=str, help='Folder name of the saved hallucinator model (default Baseline: no hallucination)')
    parser.add_argument('--data_path', default=None, type=str, help='Path of the saved class_mapping or similar_labels dictionaries')
    parser.add_argument('--sim_set_path', type=str, help='Path of the saved sim_set_sorted dictionary')
    parser.add_argument('--sim_thre_frac', default=5, type=int, help='Threshold (fractional part) for the cosine similarity to decide similar or not')
    parser.add_argument('--min_base_labels', default=1, type=int, help='Minimum number of similar base labels per label (default 1: at least one similar base label)')
    parser.add_argument('--max_base_labels', default=5, type=int, help='Maximum number of similar base labels per label (default 5: at most five similar base label)')
    parser.add_argument('--coarse_specific', default=False, type=bool, help='Use coarse-label-specific hallucinators or not')
    parser.add_argument('--n_fine_classes', default=100, type=int, help='Number of classes (base + novel)')
    parser.add_argument('--n_shot', default=1, type=int, help='Number of shot')
    parser.add_argument('--n_min', default=40, type=int, help='Minimum number of samples per training class')
    parser.add_argument('--n_top', default=5, type=int, help='Number to compute the top-n accuracy')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=3e-6, type=float, help='Learning rate')
    parser.add_argument('--learning_rate_aux', default=3e-7, type=float, help='Learning rate for the auxiliary task')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--num_epoch', default=10, type=int, help='Max number of training epochs')
    parser.add_argument('--num_epoch_per_hal', default=2, type=int, help='Number of training epochs per hallucination (not used anymore)')
    parser.add_argument('--n_iteration_aux', default=20, type=int, help='Number of iteration of the auxiliary task (not used anymore)')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--z_dim', default=100, type=int, help='Dimension of the input noise to the GAN-based hallucinator')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early-stopping mechanism')
    parser.add_argument('--GAN', action='store_true', help='Use GAN-based hallucinator if present')
    parser.add_argument('--GAN2', action='store_true', help='Use GAN2-based hallucinator if present')
    parser.add_argument('--VAEGAN2', action='store_true', help='Use semantics-guided hallucinator if present')
    parser.add_argument('--fix_seed', action='store_true', help='Pick the first n_shot shot for each novel class if present')
    parser.add_argument('--train_hal_from_scratch', action='store_true', help='Train hallucinator from scratch if present (not used anymore)')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    
    args = parser.parse_args()
    train(args)
    inference(args)

# Use base classes to train the feature extractor
def train(args):
    print('============================ train ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        if args.GAN:
            net = FSL_PN_GAN(sess,
                  model_name=args.model_name,
                  ## save FSL results and models under the hallucinator's folder
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  fc_dim=args.fc_dim,
                  z_dim=args.z_dim,
                  n_fine_class=args.n_fine_classes,
                  l2scale=args.l2scale)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                  model_name=args.model_name,
                  ## save FSL results and models under the hallucinator's folder
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  fc_dim=args.fc_dim,
                  z_dim=args.z_dim,
                  n_fine_class=args.n_fine_classes,
                  l2scale=args.l2scale)
        elif args.VAEGAN2:
            net = FSL_PN_VAEGAN2(sess,
                  model_name=args.model_name,
                  ## save FSL results and models under the hallucinator's folder
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  word_emb_path=args.word_emb_path,
                  emb_dim=args.emb_dim,
                  fc_dim=args.fc_dim,
                  z_dim=args.z_dim,
                  n_fine_class=args.n_fine_classes,
                  l2scale=args.l2scale)
        else:
            net = FSL_PN(sess,
                  model_name=args.model_name,
                  ## save FSL results and models under the hallucinator's folder
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_fine_classes,
                  l2scale=args.l2scale)
        all_vars, trainable_vars, all_regs = net.build_model()
        if args.GAN or args.GAN2:
            res = net.train(train_novel_path=os.path.join(args.result_path, args.extractor_name, 'train_novel_feat'),
                            train_base_path=os.path.join(args.result_path, args.extractor_name, 'train_base_feat'),
                            hal_from=os.path.join(args.result_path, args.hallucinator_name, 'models_hal_pro'),
                            #mlp_from=os.path.join(args.result_path, args.extractor_name, 'models'),
                            data_path=args.data_path,
                            sim_set_path=args.sim_set_path,
                            sim_thre_frac=args.sim_thre_frac,
                            min_base_labels=args.min_base_labels,
                            max_base_labels=args.max_base_labels,
                            coarse_specific=args.coarse_specific,
                            n_shot=args.n_shot,
                            n_min=args.n_min,
                            n_top=args.n_top,
                            bsize=args.bsize,
                            learning_rate=args.learning_rate,
                            learning_rate_aux=args.learning_rate_aux,
                            num_epoch=args.num_epoch,
                            num_epoch_per_hal=args.num_epoch_per_hal,
                            n_iteration_aux=args.n_iteration_aux,
                            fix_seed=args.fix_seed,
                            train_hal_from_scratch=args.train_hal_from_scratch)
        elif args.VAEGAN2:
            res = net.train(train_novel_path=os.path.join(args.result_path, args.extractor_name, 'train_novel_feat'),
                            train_base_path=os.path.join(args.result_path, args.extractor_name, 'train_base_feat'),
                            hal_from=os.path.join(args.result_path, args.hallucinator_name, 'models_hal_pro_enc'),
                            #mlp_from=os.path.join(args.result_path, args.extractor_name, 'models'),
                            data_path=args.data_path,
                            sim_set_path=args.sim_set_path,
                            sim_thre_frac=args.sim_thre_frac,
                            min_base_labels=args.min_base_labels,
                            max_base_labels=args.max_base_labels,
                            coarse_specific=args.coarse_specific,
                            n_shot=args.n_shot,
                            n_min=args.n_min,
                            n_top=args.n_top,
                            bsize=args.bsize,
                            learning_rate=args.learning_rate,
                            learning_rate_aux=args.learning_rate_aux,
                            num_epoch=args.num_epoch,
                            num_epoch_per_hal=args.num_epoch_per_hal,
                            n_iteration_aux=args.n_iteration_aux,
                            fix_seed=args.fix_seed,
                            train_hal_from_scratch=args.train_hal_from_scratch)
        else:
            res = net.train(train_novel_path=os.path.join(args.result_path, args.extractor_name, 'train_novel_feat'),
                            train_base_path=os.path.join(args.result_path, args.extractor_name, 'train_base_feat'),
                            hal_from=os.path.join(args.result_path, args.hallucinator_name, 'models_hal_pro'),
                            #mlp_from=os.path.join(args.result_path, args.extractor_name, 'models'),
                            data_path=args.data_path,
                            sim_set_path=args.sim_set_path,
                            sim_thre_frac=args.sim_thre_frac,
                            min_base_labels=args.min_base_labels,
                            max_base_labels=args.max_base_labels,
                            coarse_specific=args.coarse_specific,
                            n_shot=args.n_shot,
                            n_min=args.n_min,
                            n_top=args.n_top,
                            bsize=args.bsize,
                            learning_rate=args.learning_rate,
                            learning_rate_aux=args.learning_rate_aux,
                            num_epoch=args.num_epoch,
                            num_epoch_per_hal=args.num_epoch_per_hal,
                            n_iteration_aux=args.n_iteration_aux,
                            train_hal_from_scratch=args.train_hal_from_scratch)
    np.save(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'), res)
    
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
    
    # Plot learning curve
    results = np.load(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'results.npy'))
    fig, ax = plt.subplots(1, 2, figsize=(15,6))
    ax[0].plot(range(1, len(results[0])+1), results[0], label='Training error')
    ax[0].set_xticks(np.arange(1, len(results[0])+1))
    ax[0].set_xlabel('Training epochs', fontsize=16)
    ax[0].set_ylabel('Cross entropy', fontsize=16)
    ax[0].legend(fontsize=16)
    ax[1].plot(range(1, len(results[1])+1), results[1], label='Training accuracy')
    ax[1].set_xticks(np.arange(1, len(results[1])+1))
    ax[1].set_xlabel('Training epochs', fontsize=16)
    ax[1].set_ylabel('Accuracy', fontsize=16)
    ax[1].legend(fontsize=16)
    plt.suptitle('Learning Curve', fontsize=20)
    fig.savefig(os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'learning_curve.jpg'),
                bbox_inches='tight')
    plt.close(fig)

# Inference
def inference(args):
    print('============================ inference ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        if args.GAN:
            net = FSL_PN_GAN(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      n_fine_class=args.n_fine_classes)
        elif args.GAN2:
            net = FSL_PN_GAN2(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      n_fine_class=args.n_fine_classes)
        elif args.VAEGAN2:
            net = FSL_PN_VAEGAN2(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      word_emb_path=args.word_emb_path,
                      emb_dim=args.emb_dim,
                      fc_dim=args.fc_dim,
                      z_dim=args.z_dim,
                      n_fine_class=args.n_fine_classes)
        else:
            net = FSL_PN(sess,
                      model_name=args.model_name,
                      result_path=os.path.join(args.result_path, args.hallucinator_name),
                      fc_dim=args.fc_dim,
                      n_fine_class=args.n_fine_classes)
        net.build_model()
        net.inference(test_novel_path=os.path.join(args.result_path, args.extractor_name, 'test_novel_feat'),
                      test_base_path=os.path.join(args.result_path, args.extractor_name, 'test_base_feat'),
                      gen_from=os.path.join(args.result_path, args.hallucinator_name, args.model_name, 'models'),
                      out_path=os.path.join(args.result_path, args.hallucinator_name, args.model_name),
                      n_top=args.n_top,
                      bsize=args.bsize)

if __name__ == '__main__':
    main()