This is the demo code of the [Product-based Neural Network for User Response Prediction](https://arxiv.org/abs/1611.00144).

### Data

Follow the instructions of [make-ipinyou-data](https://github.com/Atomu2014/make-ipinyou-data) and update the soft link `data`.

For example:
```
XXX/product-nets$ ln -sfn XXX/make-ipinyou-data/2997 data
```

### Model

LR, FM, FNN, CCPM and PNN are all implemented in `models.py`, based on TensorFlow.

You can train any of the models in `main.py`.

For any questions, please report the issues or contact [Yanru Qu](http://apex.sjtu.edu.cn/members/kevinqu@apexlab.org).

