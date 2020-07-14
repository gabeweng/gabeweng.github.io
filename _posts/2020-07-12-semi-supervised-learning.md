---
title: "Semi-Supervised Learning in Computer Vision"
date: 2020-07-12T14:24-00:00
categories:
  - semi-supervised-learning
excerpt: A comprehensive overview of recent semi-supervised learning methods in Computer Vision
header:
  og_image: /images/ssl-pseudo-label.png
  teaser: /images/ssl-pseudo-label.png
classes: wide
---

Semi-supervised learning methods for Computer Vision have been progressing rapidly in the past few years. Recent state-of-the-art methods are simplifying previous work in terms of architecture and loss function or proposing hybrid methods by combining different formulations.

In this post, I will explain the key ideas of these recent methods for semi-supervised learning through diagrams.


## **1. Self-Training**
In this semi-supervised formulation, a model is trained on labeled data and used to predict pseudo-labels for the unlabeled data. The model is trained on both ground truth labels and pseudo-labels simultaneously.
![](/images/ssl-self-training.png){: .align-center}  

### a. Pseudo-label
[Dong-Hyun Lee](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf) proposed a very simple and efficient formulation called "Pseudo-label" in 2013.

The idea is to train a <span style="background-color: #e8f5e9;">model</span> simultaneously on a batch of both labeled and unlabeled images. The <span style="background-color: #e8f5e9;">model</span> is trained on labeled images in usual supervised manner with a cross-entropy loss. The same model is used to get predictions for a batch of unlabeled images and the <span style="background-color: #fce4ec;">maximum confidence class</span> is used as the <span style="background-color: #f3e5f5;">pseudo-label</span>. Then, cross-entropy loss is calculated for the unlabeled images as well.

![](/images/ssl-pseudo-label.png){: .align-center}

The overall loss is the weighted sum of the labeled and unlabeled loss terms.

$$
L = L_{labeled} + \alpha_{t} * L_{unlabeled}
$$

To make sure the model has learned enough from the labeled data, the $$\alpha_t$$ term is set to 0 during initial training steps and gradually increased up to 600 training steps and then kept constant.
![](/images/ssl-pseudolabel-alpha-increase.png){: .align-center}


### b. Noisy Student
[Xie et al.](https://arxiv.org/abs/1911.04252) proposed a semi-supervised method inspired by Knowledge Distillation called "Noisy Student" in 2019.

The key idea is to train two models called <span style="background-color: #e8f5e9;">"Teacher"</span> and <span style="background: #fff3e0;">"Student"</span>. The <span style="background-color: #e8f5e9;">teacher model</span> is first trained on the labeled images and then it is used to infer the pseudo-labels for the unlabeled images. These pseudo-labels can either be soft-label or converted to hard-label by <span style="background-color: #fce4ec;">taking the most confident class</span>. Then, the labeled and unlabeled images are combined together and a <span style="background-color: #fff3e0;">student model</span> is trained on this combined data. Input noise is injected to student model by using RandAugment to augment the images. Also, model noise such as Dropout and Stochastic Depth are used in the student model.

![](/images/ssl-noisy-student.png){: .align-center}

Once a <span style="background-color: #fff3e0;">student model</span> is trained, it becomes the new <span style="background-color: #e8f5e9;">teacher</span> and this process is repeated for three iterations.

## **2. Consistency Regularization**
This paradigm uses the idea that <span style="background-color: #e8f5e9;">model</span> predictions on an unlabeled image should remain the same even after adding noise. We could use input noise such as Image Augmentation and Gaussian noise. Noise can also be added to the architecture itself using techniques such as Dropout and Stochastic Depth.  

![Consistency Regularization Concept](/images/fixmatch-unlabeled-augment-concept.png){: .align-center}  

### a. π-model
This model was proposed by [Laine et al.](https://arxiv.org/abs/1610.02242) in a conference paper at ICLR 2017.

The key idea is to create two random augmented versions of an image for both labeled and unlabeled data. Then, a <span style="background-color: #e8f5e9;">model with dropout module</span> is used to predict the label of both these images. The <span style="background-color: #ede7f6;">square difference</span> of these two <span style="background-color: #e3f2fd;">predictions</span> is used as a <span style="background-color: #ede7f6;">consistency loss</span>. For labeled images, we also calculate the <span style="background-color: #e0f2f1;">cross-entropy loss</span>. The final loss is a weighted sum of these two loss terms. A weight <span style="background-color: #eeeeee;">w(t)</span> is applied to decide how much the consistency loss contributes in the overall loss.  

![](/images/ssl-pi-model.png){: .align-center}

### b. Temporal Ensembling  
This method was also proposed by [Laine et al.](https://arxiv.org/abs/1610.02242) in the same paper as the pi-model. This method modifies the π-model by leveraging the Exponential Moving Average(EMA) of predictions.    

The key idea is to store the <span style="background-color: #fff3e0;">exponential moving average</span> of past predictions as one view. To get another view, we augment the image as usual and a <span style="background-color: #e8f5e9;">model</span> with dropout is used to predict the label. The <span style="background-color: #ede7f6;">square difference</span> of <span style="background-color: #e3f2fd;">current prediction</span> and <span style="background-color: #fff3e0;">EMA prediction</span> is used as a <span style="background-color: #ede7f6;">consistency loss</span>. For labeled images, we also calculate the <span style="background-color: #e0f2f1;">cross-entropy loss</span>. The final loss is a weighted sum of these two loss terms. A weight <span style="background-color: #eeeeee;">w(t)</span> is applied to decide how much the consistency loss contributes in the overall loss.  

![](/images/ssl-temporal-ensembling.png){: .align-center}

### c. Mean Teacher
This method was proposed by [Tarvainen et al.](https://arxiv.org/abs/1703.01780 "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"). The general idea is similar to Temporal Ensembling but it uses Exponential Moving Average(EMA) of the model parameters instead of predictions.  

The key idea is to have two models called <span style="background-color: #e8f5e9;">"Student"</span> and <span style="background-color: #ffebee;">"Teacher"</span>. The <span style="background-color: #e8f5e9;">student</span> model is a regular model with dropout. And the <span style="background-color: #ffebee;">teacher</span> model has the same architecture as the <span style="background-color: #e8f5e9;">student</span> model but its weights are set using an <span style="background-color: #ffebee;">exponential moving average</span> of the weights of <span style="background-color: #e8f5e9;">student</span> model. For a labeled or unlabeled image, we create two random augmented versions of the image. Then, the <span style="background-color: #e8f5e9;">student</span> model is used to predict <span style="background-color: #e3f2fd;">label distribution</span> for first image. And, the <span style="background-color: #ffebee;">teacher</span> model is used to predict the <span style="background-color: #e3f2fd;">label distribution</span> for the second augmented image. The <span style="background-color: #ede7f6;">square difference</span> of these two <span style="background-color: #e3f2fd;">predictions</span> is used as a <span style="background-color: #ede7f6;">consistency loss</span>. For labeled images, we also calculate the <span style="background-color: #e0f2f1;">cross-entropy loss</span>. The final loss is a weighted sum of these two loss terms. A weight <span style="background-color: #eeeeee;">w(t)</span> is applied to decide how much the consistency loss contributes in the overall loss.  

![](/images/ssl-mean-teacher.png){: .align-center}

### d. Virtual Adversarial Training
This method was proposed by [Miyato et al.](https://arxiv.org/abs/1704.03976 "Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning"). It uses the concept of adversarial examples for consistency regularization.  

The key idea is to generate an adversarial transformation of an image that will change the model output. To do so, first, an image is taken and an adversarial variant of it is created such that the KL-divergence between the model output for the original image and the adversarial image is maximized. 

Then we proceed as previous methods. We take a labeled/unlabeled image and take its adversarial example generated in previous step as the second view. Then, the same <span style="background-color: #e8f5e9;">model</span> is used to predict <span style="background-color: #e3f2fd;">label distributions</span> for both images. The <span style="background-color: #ede7f6;">KL-divergence</span> of these two <span style="background-color: #e3f2fd;">predictions</span> is used as a <span style="background-color: #ede7f6;">consistency loss</span>. For labeled images, we also calculate the <span style="background-color: #e0f2f1;">cross-entropy loss</span>. The final loss is a weighted sum of these two loss terms. A weight <span style="background-color: #eeeeee;">$$\alpha$$</span> is applied to decide how much the consistency loss contributes in the overall loss. 


![](/images/ssl-virtual-adversarial-training.png){: .align-center}

### e. Unsupervised Data Augmentation
This method was proposed by [Xie et al.](https://arxiv.org/abs/1904.12848 "Unsupervised data augmentation for consistency training"). 

The key idea is to create an augmented version of a unlabeled image using AutoAugment. Then, a same <span style="background-color: #e8f5e9;">model</span> is used to predict the label of both these images. The <span style="background-color: #ede7f6;">KL-divergence</span> of these two <span style="background-color: #e3f2fd;">predictions</span> is used as a <span style="background-color: #ede7f6;">consistency loss</span>. For labeled images, we only calculate the <span style="background-color: #e0f2f1;">cross-entropy loss</span>. The final loss is a weighted sum of these two loss terms. A weight <span style="background-color: #eeeeee;">w(t)</span> is applied to decide how much the consistency loss contributes in the overall loss. 

![](/images/ssl-unsupervised-data-augmentation.png){: .align-center}

## **3. Hybrid Methods**

### a. MixMatch
This method was proposed by [Berthelot et al.](https://arxiv.org/abs/1905.02249 "Mixmatch: A holistic approach to semi-supervised learning").  

To understand this method, let's take a walk through each of the steps.  
  
i. For the labeled image, we create an augmentation of it. For the unlabeled image, we create K augmentations and get the model <span style="background-color: #ffebee;">predictions</span> on all K-images. Then, the <span style="background-color: #ffebee;">predictions</span> are <span style="background-color: #e0f7fa;">averaged</span> and <span style="background-color: #e3f2fd;">temperature scaling</span> is applied to get a final pseudo-label. This pseudo-label will be used for all the K-augmentations.  

![](/images/ssl-mixmatch-part-1.png){: .align-center}

ii. The batches of augmented labeled and unlabeled images are combined and the whole group is shuffled. Then, the first N images of this group are taken as $$W_L$$, and remaining M images are taken as $$W_U$$.  

![](/images/ssl-mixmatch-part-2.png){: .align-center}

iii. Now, Mixup is applied between the augmented labeled batch and group $$W_L$$. Similarly, mixup is applied between the M augmented unlabeled group and the $$W_U$$ group. Thus, we get labeled and unlabeled group.  

![](/images/ssl-mixmatch-part-3.png){: .align-center}

iv. Now, for the labeled group, we take model predictions and compute <span style="background-color: #e0f2f1;">cross-entropy loss</span> with the ground truth mixup labels. Similarly, for the unlabeled group, we compute model predictions and compute <span style="background-color: #ede7f6;">mean square error(MSE) loss</span> with the mixup pseudo labels. A weighted sum is taken of these two terms with $$\lambda$$ weighting the MSE loss.  

![](/images/ssl-mixmatch-part-4.png){: .align-center}

<!-- ### b. ReMixMatch-->

### b. FixMatch
This method was proposed by [Sohn et al.](https://arxiv.org/abs/2001.07685 "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence") and combines together pseudo-labeling and consistency regularization while vastly simplifying the overall method.    

As seen, we train a supervised model on our labeled images with cross-entropy loss. For each unlabeled image, <span style="background-color:#efdcd5">weak augmentation</span> and <span style="background-color: #e8f5e9">strong augmentations</span> are applied to get two images. The <span style="background-color:#efdcd5;">weakly augmented image</span> is passed to our model and we get prediction over classes. The probability for the most confident class is compared to a <span style="background-color: #fce4ec">threshold</span>. If it is above the <span style="background-color: #fce4ec;">threshold</span>, then we take that class as the ground label i.e. <span style="background-color: #f3e5f5;">pseudo-label</span>. Then, the <span style="background-color: #e8f5e9">strongly augmented</span> image is passed through our model to get a prediction over classes. This <span style="background-color: #e1f5fe;">probability distribution</span> is compared to ground truth <span style="background-color: #f3e5f5;">pseudo-label</span> using cross-entropy loss. Both the losses are combined and the model is tuned.

![](/images/fixmatch-pipeline.png){: .align-center}

## Comparison of Methods  
Here is a high-level summary of differences between all the above mentioned methods.  

|Method Name|Year|Unlabeled Loss|Augmentation|
|---|---|---|---|
|Pseudo-label|2013|Cross-Entropy|Random|
|π-model|2016|MSE|Random|
|Temporal Ensembling|2016|MSE|Random|
|Mean Teacher|2017|MSE|Random|
|Virtual Adversarial Training(VAT)|2017|KL-divergence|Adversarial transformation|
|Unsupervised Data Augmentation(UDA)|2019|KL-divergence|AutoAugment|
|MixMatch|2019|MSE|Random|
|Noisy Student|2019|Cross-Entropy|RandAugment|
|FixMatch|2020|Cross-Entropy|CTAugment / RandAugment|

## Common Evaluation Datasets  
To evaluate the performance of these semi-supervised methods, these dataset are commonly used.

|Dataset|Classes|Image Size|Train|Validation|Unlabeled|Remarks|
|---|---|---|---|---|---|
|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)|10|32*32|50,000|10,000|-|Subset of tiny images dataset|
|[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)|100|32*32|50,000|10,000|-|Subset of tiny images dataset|
|[STL-10](http://ai.stanford.edu/~acoates/stl10/)|10|96*96|5000|8000|1,00,000|Subset of ImageNet|
|[SVHN](http://ufldl.stanford.edu/housenumbers/)|10|32*32|73,257|26,032|5,31,131|Google Street View House Numbers|
|[ILSVRC-2012](https://www.tensorflow.org/datasets/catalog/imagenet2012)|1000|vary|1.2 million|150,000|1,50,000|Subset of ImageNet|


<!-- Part 2: Classic methods
- S4L
- Ladder Network
- Bad GAN
- Interpolation Consistency Training(ICT) for SSL
- RealMix
- Stochastic Weight Averaging (SWA)
- EnAET
- Dual Student
- CC-GAN²
- Semi-supervised self-training of object detection models.  [pseudolabeling]
-->

## Citation Info (BibTex)
If you found this blog post useful, please consider citing it as:
```
@misc{chaudhary2020semisupervised,
  title   = {Semi-Supervised Learning in Computer Vision},
  author  = {Amit Chaudhary},
  year    = 2020,
  note    = {\url{https://amitness.com/2020/07/semi-supervised-learning/}}
}
```

## References
- Dong-Hyun Lee, ["Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks"](http://deeplearning.net/wp-content/uploads/2013/03/pseudo_label_final.pdf)
- Qizhe Xie et al., ["Self-training with Noisy Student improves ImageNet classification"](https://arxiv.org/abs/1911.04252)
- Samuli Laine et al., ["Temporal Ensembling for Semi-Supervised Learning"](https://arxiv.org/abs/1610.02242)
- Antti Tarvainen et al., ["Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results"](https://arxiv.org/abs/1703.01780)
- Takeru Miyato et al., ["Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning"](https://arxiv.org/abs/1704.03976)
- Qizhe Xie et al., ["Unsupervised data augmentation for consistency training"](https://arxiv.org/abs/1904.12848)
- Hongyi Zhang, et al. ["mixup: Beyond Empirical Risk Minimization"](https://arxiv.org/abs/1710.09412)
- David Berthelot et al., ["Mixmatch: A holistic approach to semi-supervised learning"](https://arxiv.org/abs/1905.02249)
- David Berthelot et al., ["ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring"](https://arxiv.org/abs/1911.09785)
- Kihyuk Sohn et al., ["FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"](https://arxiv.org/abs/2001.07685)
