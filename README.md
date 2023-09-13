# Handwritten Digit Recognition
I split the data from the MNIST data set into a training set of 50,000 inputs, and test set of 10,000 inputs. I figured that 3 Convolutional layers would be enough to extract key features from digits, and used ReLU activations, followed by 2D Max Pooling and finally 3 Dense layers and a Softmax activation to extract the prediction. 
This gave a 98.8% accuracy on the test set, but it didn’t quite perform as well on my own writing, which pointed to overfitting. I decided to add Dropout layers after both the Convolution layers and the Dense layers and the improvement was noticeable next time I tried to draw my own digits using OpenCV:


## [Watch the demo](https://drive.google.com/file/d/1On99POBdZ6D1kQfd-ocvgR5ROgtYqOMz/view?usp=sharing)

Resources:
* MNIST dataset
* [Analysis on the Dropout Effect in Convolutional Neural Networks](http://mipal.snu.ac.kr/images/1/16/Dropout_ACCV2016.pdf)
* [Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/pdf/1207.0580.pdf)
