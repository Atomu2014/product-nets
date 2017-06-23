# Product-based Neural Networks for User Response Prediction

This repository contains the demo code of the paper 
[Product-based Neural Network for User Response Prediction](https://arxiv.org/abs/1611.00144) 
and other baseline models, implemented with ``tensorflow``. 
And this paper has been published on ICDM2016.

## An a-bit-long Introduction to User Response Prediction

User response prediction takes fundamental and crucial role in today's business, e.g. Click-Through-Rate Estimation, Recommender System, Link Prediciton. 
Different from traditional machine learning tasks, user response prediction always requires ``categorical features`` grouped by different ``fields``, 
which we call ``multi-field categorical data``. 
Here is an example, ad. request={'weekday': 3, 'hour': 18, 'IP': 255.255.255.255, 'domain': xxx.com, 'advertiser': 2997, 'click': 1}. 
In practice, these categorical features are usually one-hot encoded or hashed for training.

This representation results in some problems in training, 
e.g. Sparse Input (Large Feature Space) and Complex Local/Global Dependency.

Traditional methods include ``Logistic Regression``, ``Matrix Factorization``, etc. 
These shallow models are easy to train and deploy, however restricted in capacity. 
GBDT provides another solution which can go very deep, however, however, 
sometimes  it also requires learning good representations of features. 
Thus we turn to deep learning for an end-to-end, high capacity, and scalable model 
to learning good representations of multi-field categorical features, capture local/global dependencies, 
and further improve prediction accuracy.

[deep-ctr](https://github.com/wnzhang/deep-ctr) is an attempt to utilize deep learning in solving user response prediction problems, 
which proposes an embedding learning mechanism to deal with multi-field categorical data. 
Related works include Convolutional Click Prediction Model (CCPM), 
which uses a CNN to conduct CTR estimation. 
There are also RNN-based methods which model a sequence for future prediction, 
(in this paper we focus on non-sequential scenarios, 
and non-sequential models can be easily combined with sequential ones).

We should point out that, shallow models are popular in engineering and contests for decades because of their simplicity. 
And a lot of brilliant works have been done to complete and improve these models. 
However, these models are restricted in capacity, 
which means they can hardly capture high-level latent patterns and 
do little contributions to learning feature representations. 
One further concern is the giant data volume.

Knowledge in this field can be grouped by shallow patterns, or deep patterns.
We can say shallow patterns have been well studied, and solved by shallow models.
But we still know little about the deep ones, e.g. high-order feature combination.
Even though we do not know which group dominates, the shallow or the deep, 
explorations are needed to broaden the way.

Potential researching issues that will be mostly concerned in the near future include but not limited to:

- Learning Representation (multi-field categorical data)

- Local/Global Dependency (deep learning)

- Efficient Training (factorial concern)

- End-to-End (to replace man-made features)

For this purpose, we propose product-nets as the author's first attempt of building DNN models on this field. 
(And sooner we will release an extended version.) 
Discussion about features, models, and training are welcome, 
and please contact [Yanru Qu](http://apex.sjtu.edu.cn/members/kevinqu@apexlab.org) for any questions and help.

## How to Use

For simplicity, we provide iPinYou dataset at [make-ipinyou-data](https://github.com/Atomu2014/make-ipinyou-data). Follow the instructions and update the soft link `data`.

For example:
```
XXX/product-nets$ ln -sfn XXX/make-ipinyou-data/2997 data
```

Besides, we build a repository on github serving as a benchmark in our Lab [APEX-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets). 
This repository contains detailed data processing, feature engineering, data storage/buffering/access. 
And for better I/O performance, this benchmark provides hdf5 APIs.
Currently we provide download links of two large scale ad-click datasets (already processed), iPinYou and Criteo. 
Movielens, Netflix, and Yahoo Music datasets will be updated later.

This code is written in python 2.7, numpy, scipy and tensorflow are required. 
LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow. 
You can train any of the models in `main.py` and set parameters via a dict.

More models and mxnet implementation will be released in the extended version.

## Practical Issues

In this section I select some discussions through my email to share.

1) Sparse Regularization (L2)

Some of you may use L2 regularization to control over-fitting. 
However, in user response prediction, traditional regularization may not be proper because of sparse input.
For example, you have 1 million input features with only 100 are non-zero (recall one-hot encoding).
Simply adding penalty to all parameters means you have to update 1 million weights in one batch,
but actually most of the parameters do not contribute to the objective.
Thus traditional regularization is not reasonable, and is expensive.

For this reason, we suggest sparse regularization, i.e. you penalize the parameters depend on your input.
The difference can be formulated by: ``x`` is sparse input, ``w`` is the weight matrix, 
instead of penalize on ``w``, we penalize on ``xw``. 
Because ``x`` is sparse and one hot encoded, 
``xw`` is equivalent to select some parameters whose input is non-zero.

2) Initialization, Small Learning Rate and Adaptive Learning Rate

We suggest initialize weights with small random numbers, 
e.g. Normal(0, 0.01), Uniform(-0.01, 0.01), or Xavier initialization.
If you choose a random distribution like gaussian, 
the principle is to make variance small enough, 
and there are some papers about network initialization. 
In deed, small random initialization is always a good choice. 
And when the numbers are small, different distributions converge to same level.
Xavier is actually using uniform or normal distribution to produce small numbers,
but it is designed to cooperate with fin-in and fan-out and produce unit variance output.
Unit variance output has a lot of advantages.

Considering large data volume, small learning rate is also recommended, 
because you have to go through the training set at least once. 
In statistical machine learning, taking linear model as an example, 
the learning rate should be less than the reciprocal of the covariance matrix's eigenvalue,
otherwise the training must diverge.
And large learning rate in MLP also leads to divergence (loss explodes to infinity).

We also suggest adaptive learning rates for better convergence, 
e.g. AddGrad, Adam, FTRL and so on. 
[This famous blog](http://sebastianruder.com/optimizing-gradient-descent/) 
compares most main-stream adaptive algorithms, and currently adam is empirically a good choice.
FTRL is published by google and performs good in on-line scenarios.
Adam can be viewed as the generalization of AdaGrad and RMSProp, 
and you can convert adam to them by changing its empirical parameters.
Tensorflow's document also admits the empirical parameters may not be a good setting for all problems.

But remember, adaptive algorithms only speed up, but dose not guarantee better performance 
because they do not follow gradients' directions.
Some experiments declare that adam's performance is slightly lower than SGD on some problems, 
even though it converges much faster.
And small batch size is also recommended (try 1 if possible).

3) Feature Engineering and Data processing

Usually you need to build an index-map to convert raw data into one-hot representation.
The features usually follow a long-tail distribution, 
which means most of the features appear only several times.
Long-tailed data does little contributions to prediction because they can not be well studied,
not to mention those only appear in training set or test set.
Using all the features usually results in extremely large feature space, 
1 million dimension is still acceptable, but 1 billion is hard to handle.
We suggest drop those rarely appearing features by a threshold, 
and this will dramatically reduce the feature space without decrease of performance.

Another thing is negative sampling. Taking online ad. as an example, 
the raw data always contains much more negative samples than positive. 
A typical positive/negative ratio is 0.1%. 
Imbalance pos/neg response not only wastes computation resources, but also lags training. 
Suppose most of your mini-batches contain only negative samples, 
this won't provide reasonable gradients, and harms convergence. 
Facebook has published a paper discussing this issue, 
and keeping pos and neg samples at similar level is a good choice. 
Bt the way, do this step before building index will reduce feature dimensions.

Before training, you may still want to normalize your input.
There are two kinds of normalization, feature level and instance level.
Feature level is within one field, 
e.g. set the mean of one field to 0 and the variance to 1.
Instance level is to keep consistent between difference records,
e.g. you have a multi-value field, which has 5-100 values and the length varies. 
You can set the sum to 1 or magnitude to 1 by applying instance-wise normalization.
Whitening is not always possible, and normalization is just enough.

4) Continuous Feature, Discrete Feature, and Multi-value Feature

Different from natural singals and sensor signals, most features in User Response Prediction have discrete values (categorical features). The key difference between continuous and discrete features is, discrete features only have absolute meanings, while continuous features have both absolute and relative meanings. For example, 'male' and 'female' are only different symbols, denoting {'male': 0, 'female': 1} or {'male': 1, 'female': 0} do not change symbols' meanings. However, numbers 1, 2, 3 can not be arbitrarily encoded, i.e. a good representation should preserve the relationships among these numbers.

For discrete values, we usually build a feature map to encode symbols as IDs. And we use a dictionary-long vector to represent this feature, with 1 denoting specific value exists. However, here arises another problem when your data contains continuous and discrete values. For example, you have 'gender' and 'height'. One data example may be {'gender': 'male', 'height': 1.8}, and you may want to encode like this {'gender': [1, 0], 'height': 1.8} <=> [1, 0, 1.8]. The problem is, in some dimensions, '1' is a symbol and denotes existence, however in other dimensions, '1' is a number describing how far it is away from '0'. Even though you can apply data normalization, the gap still exists and we still need to unify these features.

Because discrete features are much more than continuous ones, we suggest discretize those continuous values using bucketing. That is, using some 'levels' to represent continuous numbers. Taking 'gender' as an example, you can set [0, 12] as 'children', [13, 18] as 'teenagers', [19, ~] as 'adults' as so on. Usually we use two principles, equal-length and equal-size. Suppose the data range is [0, 100], you can set thresholds as [20, 40, 60, 80] and this strategy produces equal-length bucketing. Or you find most of the data lie around 50 and form a gaussian distribution, then you can set thresholds as [30, 40, 50, 60, 70] to let every bucket holds same number of data.

Multi-value features are special cases of discrete features. For example, 'recently reviewed items' may have several values, like ['item2', 'item7', 'item11'], or ['item1', 'item4', 'item9', 'item13']. You can simply use multi-hot encoding, but there are several side effects. Suppose one user has reviewed 3 items, and another has reviewed 300 items, matrix-multiplication based operations will add these items up and result in huge imbalance. You may turn to data normalization to tackle this problem. Till now, there is still not a standard representation for multi-value features, and we are still working on it. 

5) Embeddings and Activation

Embedding is a linear operation, adding non-linearity to this makes no sense.
And recently, our experiments show that activation on embedding vectors will not improve performance,
thus we will remove activation function after embedding layer in the extended version.
Here gives some discussion.

Suppose the embedding layer is followed by a fully connected layer. 
The embedding can be represented by ``xw1``. 
The fc layer can be represented by ``s(xw1) w2``.
If ``s`` is some nonlinear activation, it can be viewed as two fc layers.
If ``s`` is identity function, it is equivalent to ``xw1 w2 = xw3``.
It seems like you are using ``w3`` as the embedding dimension followed by a nonlinear activation.
The only difference is ``w1 w2``'s rank is different from ``w3``. 
When ``w1`` ``w2``'s inner dimension is small, their product must not by full-rank.
But ``w3`` does not have this natural constraint.

Besides, we find ``relu`` performs better than ``tanh``, and ``sigmoid`` has worst convergence when using neural networks.
