# Product-based Neural Networks for User Response Prediction

This repository contains the demo code of the paper [Product-based Neural Network for User Response Prediction](https://arxiv.org/abs/1611.00144) and other baseline models, implemented with ``tensorflow``. ANd this paper has been published on ICDM2016.

### An a-bit-long Introduction to User Response Prediction and Some Suggestions

User response prediction is a large family of researching tasks taking fundamental and crucial role in today's business, e.g. Click-Through-Rate Estimation, Recommender System, Link Prediciton. Different from traditional machine learning tasks, user response prediciton always requires ``categorical features`` with different ``fields``, which we call ``multi-field categorical data``. Here is an example, ad. request={'weekday': 3, 'hour': 18, 'IP': 255.255.255.255, 'domain': xxx.com, 'advertiser': 2997, 'click': 1}. In practice, these categorical features are usually one-hot encoded or hashed for training.

The main challenges in user response prediction are:

- Sparse Input (Large Feature Space)

- Categorical Features

- Large Data Volume

- Imbalance Feedback

- Complex Local/Gloabl Dependency

Traditional methods include Logistic Regression, Matrix Factorization, etc. These shallow models are easy to train and deploy, however restricted in capacity. GBDT provides another solution which can go very deep, however, GBDT sometimes also requires learning good representations of features. Thus we turn to deep learning for an end-to-end, high capacity, and scalable model for user response prediction.

[deep-ctr](https://github.com/wnzhang/deep-ctr) is an attempt to utilize deep learning in solving user response prediction problems, which proposes an embedding learning mechanism to deal with multi-field categorical data. Related works include Convolutional Click Prediction Model (CCPM), which uses a CNN to conduct CTR estimation. There are also RNN models which model record sequence for future prediction, (in this paper we only discuss non-sequencial senarios).

We should point out that, shallow models are popular in engineering and contests for decades because of their simplicity. And a lot of brilliant works have been done to improve these models. However, these models are restricted in capacity, which means they can hardly capture high-level latent patterns. Even though we do not know which factor dominates, the shallow pattern or the deep pattern, explorations are needed to broaden the way.

Potential researching issues include but not limited to:

- Learning Representation

- Local/Global Dependency

- Efficient Training

- End-to-End (to replace man-made features)

For this purpose, we propose product-nets as the author's first attempt of building DNN models on this field. (And we are still making efforts to make it better.) Discussion about features, models, and training are welcome, and please contact [Yanru Qu](http://apex.sjtu.edu.cn/members/kevinqu@apexlab.org) for any questions.

### How to Use

For simplicity, we provide iPinYou dataset at [make-ipinyou-data](https://github.com/Atomu2014/make-ipinyou-data). Follow the instructions and update the soft link `data`.

For example:
```
XXX/product-nets$ ln -sfn XXX/make-ipinyou-data/2997 data
```

Besides, we build a repository on github as benchmarks in our Lab [APEX-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets). This repository contains detailed data processing, feature engineering, data storage/buffering/access. Currently we provide download links of two large scale ad-click datasets, iPinYou and Criteo. Movielens, Netflix, and Yahoo Music datasets will be updated later.

This code is written in python 2.7, numpy, scipy and tensorflow are required. LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow. You can train any of the models in `main.py` and the parameters are all presented in a dict.

More models and mxnet implementation will be released in the extended version.
