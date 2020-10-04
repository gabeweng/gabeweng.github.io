---
title: "Knowledge Transfer in Self Supervised Learning"
date: 2020-10-04T00:00-00:00
categories:
  - self-supervised-learning
permalink: /knowledge-transfer/
classes: wide
excerpt: A general framework to transfer knowledge from deep self-supervised models to shallow task-specific models
header:
  og_image: /images/kt-step-3.png
  teaser: "/images/kt-step-3.png"
---

Self Supervised Learning is an interesting research area where the goal is to learn rich representations from unlabeled data without any human annotation. 

This can be achieved by creatively formulating a problem such that you use parts of the data itself as labels and try to predict that. Such formulations are called pretext tasks.

![](/images/kt-pretext-tasks.png){:.align-center}  

By pre-training on the pretext task, the hope is that the model will learn useful representations. Then, we can finetune the model to downstream tasks such as image classification, object detection and semantic segmentation with only a small set of labeled training data.  

## Challenge of evaluating representations  
We learned how pretext tasks can give us representations. But, this poses a question:
> How to determine how good a learned representations is?

Currently, the standard way to gauge the representations is to evaluate it on a set of standard tasks and benchmark datasets.
- Linear classification on ImageNet using frozen features
- Image Classification using only 1% to 10% of labels on ImageNet
- Transfer Learning: Object Detection on PASCAL VOC

We can see that the above evaluation method requires us to use the same model architecture for both pre-text and target task.  

![](/images/kt-pretext-target-challenge.png){:.align-center}  

This poses some interesting challenges:
1. For pre-text task, our goal is to learn on large scale unlabeled dataset and thus deeper models(e.g. ResNet) would help us learn better representations. 

    But, for downstream tasks, we would prefer shallow models(e.g. AlexNet) for actual applications. This limits our ability to use large scale datasets as we have to consider the final task into account as well.  

2. It's harder to fairly compare which pre-text task is better if some methods used simpler architecture while other methods used deeper architecture.

## Knowledge Transfer
Noroozi et al. proposed a simple idea to tackle this issue in their paper ["Boosting Self-Supervised Learning via Knowledge Transfer"](https://arxiv.org/abs/1805.00385).  

### Intuition  
The authors observed that in a good representation space, semantically similar data points should be close together.  

![](/images/kt-good-vs-bad-representation.png){:.align-center}  

In regular supervised classification, the information that images are semantically similar is encoded through labels annotated by humans. A model trained on such labels would have a representation space that groups semantically similar images.  

Thus, with pre-text tasks in self-supervised learning, the objective is implicitly learning a metric that makes same category images similar and different category images dissimilar. Hence we can provide a robust estimate of the learned representation if we could encode semantically related images to the same labels in some way.

## General Framework  
The authors propose a novel framework to transfer knowledge from a deep self-supervised model to a separate shallow downstream model. You can use different model architectures for pre-text task and downstream task.  

The end-to-end process is as follows:

#### 1. Pre-text task:  
Here we choose some deep network architecture and train it on some pretext task of our choice on some dataset. We can take features from some intermediate layer after the model is trained.  

![](/images/kt-step-1.png){:.align-center}  
Figure: Training on Pre-text Task ([Source](https://arxiv.org/abs/1805.00385))
{: .text-center}

#### 2. **Clustering**:  
For all the unlabeled images in the dataset, we compute the feature vectors from pretext task model. Then, we run K-means clustering to group semantically similar images. The idea is that the cluster centers will be aligned with categories in ImageNet.

![](/images/kt-step-2.png){:.align-center}  
Figure: Clustering Features ([Source](https://arxiv.org/abs/1805.00385))
{: .text-center}

In the paper, the authors ran K-means on a single Titan X GPU for 4 hours to cluster 1.3M images into 2000 categories.  

#### 3. Pseudo-labeling  
The cluster centers is treated as the pseudo-label. We can use either the same dataset as above step or use a different dataset itself. Then, we compute the feature vectors for those images and find the closest cluster center for each image. This cluster center is used as the pseudo-label.  

![](/images/kt-step-3.png){:.align-center}  
Figure: Generating Pseudo-labels ([Source](https://arxiv.org/abs/1805.00385))
{: .text-center}

#### 4. Training on Pseudo-labels  
We take the model architecture that will be used for downstream task and train it to classify the unlabeled images into the pseudo-labels. Thus, the target architecture will learn a new representation such that it will map images that were originally close in the pre-trained feature space to close points.  

![](/images/kt-step-4.png){:.align-center}  
Figure: Re-training on pseudo-labels ([Source](https://arxiv.org/abs/1805.00385))
{: .text-center}

## How well does this framework work?
To test this idea, the authors did an experiment as described below:

### a. Increase complexity of pretext task (Jigsaw++)
To evaluate their method, the authors took an old puzzle-like pretext task called "Jigsaw" where we need to predict the permutation that was used to randomly shuffle a 3*3 square grid of image.

They extended the task by randomly replacing 0 to 2 number of tiles with tile from another random image at some random locations. This increases the difficulty as now we need to solve the problem using only the remaining patches. The new pretext task is called "Jigsaw++".

### b. Use deeper network to solve pretext task
The authors used VGG-16 to solve the pretext task and learn representations. As VGG-16 has increased capacity, it can better handle the increased complexity of "Jigsaw++" task and thus extract better representation.

### c. Transfer Knowledge back to AlexNet
The representations from VGG-16 are clustered and cluster centers are converted to pseudo-labels. Then, AlexNet is trained to classify the pseudo-labels.

### d. Finetune AlexNet on Evaluation datasets
For downstream tasks, the conv layers for the AlexNet model are initialized with weights from pseudo-label classification and the fully connected layers were randomly initialized.
The pre-trained AlexNet is then finetuned on various benchmark datasets.

### e. Results  
Using deeper network like VGG-16 leads to better representation and pseudo-labels and also better results in benchmark tasks.
#### 1. Transfer Learning on PASCAL VOC
The authors tested their method on object classification and detection on PASCAL VOC 2007 dataset and semantic segmentation on PASCAL VOC 2012 dataset. 

**Insights**:  
- Switching to a difficult task Jigsaw++ boost performance than Jigsaw.
- Knowledge transfer doesn't have significant impact when using same architecture AlexNet to solve both Jigsaw++ and downstream task.
- Training Jigsaw++ with VGG16 and using AlexNet to predict cluster gives the best performance.

|Task|Clustering|Pre-text architecture|Downstream arch.|Classification|Detection(SS)|Detection(MS)|Segmentation|
|---|---|---|---|---|---|---|---|
|Jigsaw|no|AlexNet|AlexNet|67.7|53.2|-|-|
|Jigsaw++|no|AlexNet|AlexNet|69.8|55.5|55.7|38.1|
|Jigsaw++|yes|AlexNet|AlexNet|69.9|55.0|55.8|40.0|
|Jigsaw++|yes|VGG-16|AlexNet|**72.5**|**56.5**|**57.2**|**42.6**|

#### 2. Linear Classification on ImageNet
In this, a linear classifier is trained on features extracted from AlexNet at different convolutional layers. For ImageNet, using VGG-16 and transfering knowledge to AlexNet using clustering gives a substantial boost of 2%.

![](/images/kt-imagenet-performance.png){:.align-center}  

#### 3. Non-linear classification on ImageNet
For a non-linear classifier trained on frozen features from various convolutional layers, the approach of using VGG-16 and transferring knowledge to AlexNet using clustering gives the best performance on ImageNet.  
![](/images/kt-nonlinear-result.png){:.align-center}  



## References
- Mehdi Noroozi et al., ["Boosting Self-Supervised Learning via Knowledge Transfer"](https://arxiv.org/abs/1805.00385)
- Mehdi Noroozi et al., ["Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles"](https://arxiv.org/abs/1603.09246)
- Daisuke Okanohara et al., ["A discriminative language model with pseudo-negative samples"](https://www.aclweb.org/anthology/P07-1010/)