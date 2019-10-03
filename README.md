# Semantics-guided Data Hallucination for Few-shot Visual Classification

## sem_hal
This code implements semantics-guided data hallucination as described in the paper [Semantics-guided Data Hallucination for Few-shot Visual Classification](https://ieeexplore.ieee.org/document/8803420). As described in the paper, the few-shot learning (FSL) framework considers a training dataset *D*<sub>base<sub> of base classes *C*<sub>base<sub>, each with a sufficient amount of images, and another training dataset *D*<sub>novel<sub> of novel classes *C*<sub>novel<sub>, each with only few images available during training. The goal of FSL is to build a classifier that can be used to predict the label of an unseen test image from the union of *C*<sub>novel<sub> and *C*<sub>base<sub>.

The training process can be divided into two phases. First, in the *representation learning* phase, only *D*<sub>base<sub> would be utilized. A convolutional neural network (CNN) based feature extractor is trained on *D*<sub>base<sub>.
