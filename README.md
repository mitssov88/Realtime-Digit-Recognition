# Handwritten Digit Recognition
I applied recent Deep Learning research in reducing neuron co-adaption &amp; the overfitting that results in sequential models to improve the accuracy of real-time handwritten digit recognition

Accuracy 98.8%

## Overview of implementing findings from research papers (cited below)
Initially, I wanted to implement a Deep Learning CNN for digit recognition (data from MNIST) that would achieve the highest possible accuracy on 10,000 tests using 50,000 training inputs. I trained a model with 3 Convolution layers, each with ReLU activation and Max Pooling, and finally 3 fully-connected Dense layers. This gave me a 98.8% accuracy on the testing data. Great. 

**But I noticed a problem**. The model didn’t perform nearly as well on my own real-time handwritten inputs which I integrated using OpenCV.
For inaccurate predictions, the digit predicted wasn’t even similar to the correct digit, and a 98.8% accuracy on test data but only approx. 70% accuracy on my real-time handwritten digits pointed to overfitting in the model. 
I did some research on co-adaptation of neurons and the overfitting that it yields, and decided to implement improvements discussed in research papers which involved adding Dropout layers after non-linear activation layers, specifically with a low rate after Conv. layers and a higher rate after Dense, fully- connected layers.
The improvement was substantial. Accuracy on my handwritten digits went from about 70% to almost 100%.

## [Watch the demo](https://drive.google.com/file/d/1On99POBdZ6D1kQfd-ocvgR5ROgtYqOMz/view?usp=sharing)

Resources:
* MNIST dataset
* [Analysis on the Dropout Effect in Convolutional Neural Networks](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)
* [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf)
