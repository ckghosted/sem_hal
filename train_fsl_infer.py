import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_FSL import FSL
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--learning_rate', default=1e-6, type=float, help='Learning rate')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--num_epoch', default=500, type=int, help='Max number of training epochs')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early-stopping mechanism')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    
    args = parser.parse_args()
    train(args)
    inference(args)

# Use base classes to train the feature extractor
def train(args):
    print('============================ train ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = FSL(sess,
                  model_name=args.model_name,
                  ## save FSL results and models under the hallucinator's folder
                  result_path=os.path.join(args.result_path, args.hallucinator_name),
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_fine_classes,
                  l2scale=args.l2scale)
        all_vars, trainable_vars, all_regs = net.build_model()
        res = net.train(train_novel_path=os.path.join(args.result_path, args.extractor_name, 'train_novel_feat'),
                        train_base_path=os.path.join(args.result_path, args.extractor_name, 'train_base_feat'),
                        hal_from=os.path.join(args.result_path, args.hallucinator_name, 'models_hal'),
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
                        num_epoch=args.num_epoch,
                        patience=args.patience)
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
        net = FSL(sess,
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