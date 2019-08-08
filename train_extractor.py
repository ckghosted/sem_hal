import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
from model_VGG import VGG
import os, re, glob

import argparse

import warnings
warnings.simplefilter('ignore')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='Path of train_novel, train_base, test_novel, and test_base')
    parser.add_argument('--result_path', type=str, help='Path to save all results')
    parser.add_argument('--extractor_name', type=str, help='Folder name to save extractor models and learning curves')
    parser.add_argument('--vgg16_npy_path', type=str, help='Path of the VGG16 pre-trained weights: vgg16.npy')
    parser.add_argument('--n_base_classes', default=80, type=int, help='Number of the base classes')
    parser.add_argument('--n_top', default=5, type=int, help='Number to compute the top-n accuracy')
    parser.add_argument('--bsize', default=64, type=int, help='Batch size')
    parser.add_argument('--learning_rate', default=1e-5, type=float, help='Learning rate')
    parser.add_argument('--l2scale', default=1e-3, type=float, help='L2-regularizer scale')
    parser.add_argument('--lambda_center_loss', default=0.0, type=float, help='L2-regularizer scale')
    parser.add_argument('--num_epoch', default=500, type=int, help='Max number of training epochs')
    parser.add_argument('--img_size_h', default=32, type=int, help='img_size_h: 32 for cifar-100, 64 for mini-imagenet')
    parser.add_argument('--img_size_w', default=32, type=int, help='img_size_w: 32 for cifar-100, 64 for mini-imagenet')
    parser.add_argument('--fc_dim', default=512, type=int, help='Fully-connected dimensions of the hidden layers of the MLP classifier')
    parser.add_argument('--patience', default=10, type=int, help='Patience for early-stopping mechanism')
    parser.add_argument('--debug', action='store_true', help='Debug mode if present')
    args = parser.parse_args()
    train(args)
    inference(args)
    extract(args)

# Use base classes to train the feature extractor
def train(args):
    print('============================ train ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = VGG(sess,
                  model_name=args.extractor_name,
                  result_path=args.result_path,
                  img_size_h=args.img_size_h,
                  img_size_w=args.img_size_w,
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_base_classes,
                  l2scale=args.l2scale,
                  lambda_center_loss=args.lambda_center_loss,
                  vgg16_npy_path=args.vgg16_npy_path)
        all_vars, trainable_vars, all_regs = net.build_model()
        res = net.train(train_path=os.path.join(args.data_path, 'train_base'),
                        n_top=args.n_top,
                        bsize=args.bsize,
                        learning_rate=args.learning_rate,
                        num_epoch=args.num_epoch,
                        patience=args.patience)
    np.save(os.path.join(args.result_path, args.extractor_name, 'results.npy'), res)
    
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
    results = np.load(os.path.join(args.result_path, args.extractor_name, 'results.npy'))
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
    fig.savefig(os.path.join(args.result_path, args.extractor_name, 'learning_curve.jpg'),
                bbox_inches='tight')
    plt.close(fig)

# Inference
def inference(args):
    print('============================ inference ============================')
    tf.reset_default_graph()
    with tf.Session() as sess:
        net = VGG(sess,
                  model_name=args.extractor_name,
                  result_path=args.result_path,
                  img_size_h=args.img_size_h,
                  img_size_w=args.img_size_w,
                  fc_dim=args.fc_dim,
                  n_fine_class=args.n_base_classes,
                  vgg16_npy_path=args.vgg16_npy_path)
        net.build_model()
        net.inference(test_path=os.path.join(args.data_path, 'test_base'),
                      gen_from=os.path.join(args.result_path, args.extractor_name, 'models'),
                      out_path=os.path.join(args.result_path, args.extractor_name),
                      n_top=args.n_top,
                      bsize=args.bsize)

# Extract all features
def extract(args):
    print('============================ extract ============================')    
    tf.reset_default_graph()
    ## In the extractor() function, all features, coarse_labels, and fine_labels
    ## will be collected as a dictionary, and saved into the path:
    ##     os.path.join(result_path, extractor_name, saved_filename)
    ## (e.g., '/home/cclin/few_shot_learning/hallucination_by_analogy/results/VGG_b64_lr1e5_fc512/train_novel_feat')
    with tf.Session() as sess:
        net = VGG(sess,
                  model_name=args.extractor_name,
                  result_path=args.result_path,
                  img_size_h=args.img_size_h,
                  img_size_w=args.img_size_w,
                  fc_dim=args.fc_dim,
                  vgg16_npy_path=args.vgg16_npy_path)
        net.build_model()
        net.extractor(test_path=os.path.join(args.data_path, 'train_novel'),
                      saved_filename='train_novel_feat',
                      gen_from=os.path.join(args.result_path, args.extractor_name, 'models_cnn'),
                      out_path=os.path.join(args.result_path, args.extractor_name),
                      bsize=args.bsize)
        net.extractor(test_path=os.path.join(args.data_path, 'train_base'),
                      saved_filename='train_base_feat',
                      gen_from=os.path.join(args.result_path, args.extractor_name, 'models_cnn'),
                      out_path=os.path.join(args.result_path, args.extractor_name),
                      bsize=args.bsize)
        net.extractor(test_path=os.path.join(args.data_path, 'test_novel'),
                      saved_filename='test_novel_feat',
                      gen_from=os.path.join(args.result_path, args.extractor_name, 'models_cnn'),
                      out_path=os.path.join(args.result_path, args.extractor_name),
                      bsize=args.bsize)
        net.extractor(test_path=os.path.join(args.data_path, 'test_base'),
                      saved_filename='test_base_feat',
                      gen_from=os.path.join(args.result_path, args.extractor_name, 'models_cnn'),
                      out_path=os.path.join(args.result_path, args.extractor_name),
                      bsize=args.bsize)

if __name__ == '__main__':
    main()