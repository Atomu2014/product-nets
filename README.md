# Product-based Neural Networks for User Response Prediction

``Note``: An extended version of the conference paper is https://arxiv.org/abs/1807.00311 , which is accepted by TOIS.
Compared with this simple demo, a more detailed implementation of the journal paper is at https://github.com/Atomu2014/product-nets-distributed , which has large-scale data access, multi-gpu support, and distributed training support.

``Note``: I would like to share some intersting and advanced discussions in the [extended version](https://github.com/Atomu2014/product-nets-distributed).

``Note``: Any problems, you can contact me at kevinqu16@gmail.com. Through email, you will get my rapid response.

This repository maintains the demo code of the paper 
[Product-based Neural Network for User Response Prediction](https://arxiv.org/abs/1611.00144) 
and other baseline models, implemented with ``tensorflow``. 
And this paper has been published on ICDM2016.

## Introduction to User Response Prediction

User response prediction takes a fundamental and crucial role in today's business, especially personalized recommender system and online display advertising. 
Different from traditional machine learning tasks, 
user response prediction always has ``categorical features`` grouped by different ``fields``, 
which we call ``multi-field categorical data``, e.g.: 

    ad. request={
        'weekday': 3, 
        'hour': 18, 
        'IP': 255.255.255.255, 
        'domain': xxx.com, 
        'advertiser': 2997, 
        'click': 1
    }

In practice, these categorical features are usually one-hot encoded for training. 
However, this representation results in sparsity.
Challenged by data sparsity, linear models (e.g., ``LR``), latent factor-based models (e.g., ``FM``, ``FFM``), tree models (e.g., ``GBDT``), and DNN models (e.g., ``FNN``, ``DeepFM``) are proposed.

A core problem in user response prediction is how to represent the complex feature interactions. Industrial applications prefer feature engineering and simple models. With GPU servers becoming more and more popular, it is promising to design complex models to explore feature interactions automatically. Through our analysis and experiments, we find a ``coupled gradient`` issue of latent factor-based models, and an ``insensitive gradient`` issue of DNN models.

Take FM as an example, the gradient of each feature vector is the sum over other feature vectors. Suppose two features are independent, FM can hardly learn two orthogonal feature vectors. The gradient issue of DNNs is discussed in the paper ``Failures of Gradient-based Deep Learning``. 

<!--Another interesting fact in recommendation or ctr contests is that, winning solutions usually transform discrete features into continuous or vice versa:
- Use GBDT to convert continuous features to binary ones, and feed binary features to FM.
- Use FM/DNN to convert discrete features to embeddings or interactions, and feed these features to GBDT.-->

In order to solve these issues, we propose to use product operators in DNN to help explore feature interactions. We discuss these issues in an extended paper, which is submitted to TOIS at Seq. 2017 and will be released later.
Any discussion is welcomed, please contact kevinqu16@gmail.com.

## Product-based Neural Networks

Through discussion of previous works, we think a good predictor should have a good feature extractor (to convert sparse features into dense representations) as well as a powerful classifier (e.g., DNN as universal approximator). Since FM is good at represent feature interactions, we introduce product operators in DNN. The proposed PNN models follow this architecture: an embedding layer to represent sparse features, a product layer to explore feature interactions, and a DNN classifier.

For product layer, we propose 2 types of product operators in the paper: inner product and outer product. These operators output $n(n-1)/2$ feature interactions, which are concatenated with embeddings and fed to the following fully conncted layers.

The inner product is easy to understand, the outer product is actually equivalent to projecting embeddings into a hidden space and computing the inner product of projected embeddings:

$uv^T\odot w = u^Twv$

Since there are $n(n-1)/2$ feature interactions, we propose some tricks to reduce complexity.
However, we find these tricks restrict model capacity and are unecessary.
In recent update of the code, we remove the tricks for better performance. 

In our implementation, we add the parameter ``kernel_type: {mat, vec, num}`` for outer product.
The default type is mat, and you can switch to other types to save time and memory.

A potential risk may happen in training the first hidden layer. Feature embeddings and interactions are concatenated and fed to the first hidden layer, but the embeddings and interactions have different distribution. A simple method is adding linear transformation to the embeddings to balance the distributions. ``Layer norm`` is also worth to try.

## How to Use

For simplicity, we provide iPinYou dataset at [make-ipinyou-data](https://github.com/Atomu2014/make-ipinyou-data). 
Follow the instructions and update the soft link `data`:

```
XXX/product-nets$ ln -sfn XXX/make-ipinyou-data/2997 data
```

run ``main.py``:

    cd python
    python main.py

As for dataset, we build a repository on github serving as a benchmark in our Lab 
[APEX-Datasets](https://github.com/Atomu2014/Ads-RecSys-Datasets). 
This repository contains detailed data processing, feature engineering, 
data storage/buffering/access and other implementations.
For better I/O performance, this benchmark provides hdf5 APIs.
Currently we provide download links of 4 large scale ad-click datasets (already processed),
Criteo-8day, Avazu, iPinYou-all, and Criteo Challenge. More datasets will be updated later.

This code is originally written in python 2.7, numpy, scipy and tensorflow are required. 
In recent update, we make it consistent with python 3.x. 
Thus you can use it as a start-up with any python version you like.
LR, FM, FNN, CCPM, DeepFM and PNN are all implemented in `models.py`, based on TensorFlow. 
You can train any of the models in `main.py` and configure parameters via a dict.

More models and mxnet implementation will be released in the extended version.

## Practical Issues

In this section we select some discussions from my emails and issues to share.

``Note``: 2 advanced discussions about overfitting of adam and performance gain of DNNs are presented in the [extended version](https://github.com/Atomu2014/product-nets-distributed). You are welcomed to discuss relavant problems through issues or emails.

### 1. Sparse Regularization (L2)

L2 is fundamental in controlling over-fitting.
For sparse input, we suggest sparse regularization, 
i.e. we only regularize on activated weights/neurons.
Traditional L2 regularization penalizes all parameters $\Vert w\Vert$, $w = [w_1, \dots, w_n]$ even though some inputs are zero $x_i = 0$,
which means every parameter $w_i$ will have a non-zero gradient for every training example $x$.
Sparse regularization instead penalizes on non-zero terms, $\Vert xw \Vert$. 

### 2. Initialization

Initializing weights with small random numbers is always promising in Deep Learning.
Usually we use ``uniform`` or ``normal`` distribution around 0.
An empirical choice is to set the distribution variance near $\sqrt{(1/n)}$ where n is the input dimension.
Another choice is ``xavier``, for uniform distribution, 
``xavier`` uses $\sqrt{(3/node_i)}$, $\sqrt{(3/node_o)}$, 
or $\sqrt{(6/(node_i+node_o))}$ as the upper/lower bound. 
This is to keep unit variance among different layers.

### 3. Learning Rate

For deep neural networks with a lot of parameters, 
large learning rate always causes divergence. 
Usually sgd with small learning rate has promising performance, however converges slow.
For extremely sparse input, adaptive learning rate converges much faster, 
e.g. AdaGrad, Adam, FTRL, etc.
[This blog](http://sebastianruder.com/optimizing-gradient-descent/) 
compares most of adaptive algorithms.
Even though adaptive algorithms speed up and sometimes jump out of local minimum, 
there is no guarantee for better generalization performance.
To sum up, ``Adam`` and ``AdaGrad`` are good choices. ``Adam`` converges faster than ``AdaGrad``, but is also easier to overfit.

### 4. Data Processing

Usually you need to build a feature map to convert categorical data into one-hot representation.
These features usually follow a long-tailed distribution,
resulting in extremely large feature space, e.g. IP address. 
A simple way is to remove those low frequency features by a threshold, 
which will dramatically reduce the input dimension without much decrease of performance.

For unbalance dataset, a typical positive/negative ratio is 0.1% - 1%, 
and Facebook has published a paper discussing negative down sampling. 
Negative down-sampling can speed up training, as well as reduce dimension, but requires calibration in some cases.

### 5. Normalization

There are two kinds of normalization, feature level and instance level.
Feature level is within one field, 
e.g. set the mean of one field to 0 and the variance to 1.
Instance level is to keep consistent between difference records,
e.g. you have a multi-value field, which has 5-100 values and the length varies. 
You can set the magnitude to 1 by shifting and scaling.
Besides, ``batch/weight/layer normalization`` are worth to try when network grows deeper.

### 6. Continuous/Discrete/Multi-value Feature

Most features in User Response Prediction have discrete values (categorical features). The key difference between continuous and discrete features is, only continuous features are comparable in values. For example, {``male``: 0, ``female``: 1} and {``male``: 1, ``female``: 0} are equivalent.

When the data contains both continuous and discrete values, one solution is to discretize those continuous values using bucketing. Taking 'age' as an example, you can set [0, 12] as ``children``, [13, 18] as ``teenagers``, [19, ~] as ``adults`` and so on. 

Multi-value features are special cases of discrete features. 
e.g. recently reviewed items = [``item2``, ``item7``, ``item11``], [``item1``, ``item4``, ``item9``, ``item13``]. 
This type of data is also called set data, with one key property ``permutation invariance``, which is discussed in the paper ``DeepSet``.

### 7. Activation Function

Do not use ``sigmoid`` in hidden layers, use ``tanh`` or ``relu`` instead.
And recently ``selu`` is proposed to maintain fixed point in training.

### 8. Numerical Stable Parameters

Adaptive optimizers usually requires hyperparameters for numerical stability, e.g., $\epsilon$ in ``Adam``, ``initial value`` of ``AdaGrad``. Sometimes, these parameters have large impacts on model convergence and performance.
