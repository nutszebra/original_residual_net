# What's this
Implementation of Residual Networks (ResNet) by chainer  

# Dependencies

    git clone https://github.com/nutszebra/original_residual_net.git
    cd original_residual_net
    git submodule init
    git submodule update

# How to run
    python main.py -p ./ -g 0 


# Details about my implementation
All hyperparameters and network architecture are the same as in [[1]][Paper] except for data-augmentation.  
* Data augmentation  
Train: Pictures are randomly resized in the range of [32, 36], then 32x32 patches are extracted randomly and are normalized locally. Horizontal flipping is applied with 0.5 probability.  
Test: Pictures are resized to 32x32, then they are normalized locally. Single image test is used to calculate total accuracy.  

# Cifar10 result

| network              | depth | total accuracy (%) |
|:---------------------|-------|-------------------:|
| ResNet [[1]][Paper]  | 110   | 93.57              |
| my implementation    | 110   | soon               |

<img src="https://github.com/nutszebra/original_residual_net/blob/master/loss.jpg" alt="loss" title="loss">
<img src="https://github.com/nutszebra/original_residual_net/blob/master/accuracy.jpg" alt="total accuracy" title="total accuracy">


# Reference
Deep Residual Learning for Image Recognition [[1]][Paper]

[paper]: https://arxiv.org/abs/1512.03385 "Paper"
