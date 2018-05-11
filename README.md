# Image Colorization with Generative Adversarial Networks 
Over the last decade, the process of automatic colorization had been studied thoroughly due to its vast application such as colorization of grayscale images and restoration of aged and/or degraded images. This problem is highly ill-posed due to the extremely large degrees of freedom during the assignment of color information. Many of the recent developments in automatic colorization involved images that contained a common theme throughout training, and/or required highly processed data such as semantic maps as input data. In our approach, we attempted to fully generalize this procedure using a conditional Deep Convolutional Generative Adversarial Network (DCGAN). The network is trained over datasets that are publicly available such as CIFAR-10 and Places365. The results of the generative model and tradition deep neural networks are compared. The colorization is done using two convolutional networks to as suggested by [Pix2Pix](https://github.com/phillipi/pix2pix).

The network is trained on the datasets [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Places365](http://places2.csail.mit.edu). Some of the results from Places365 dataset are [shown here](#places365-results)

## Prerequisites
- Linux
- Tensorflow 1.4
- Keras
- NVIDIA GPU (12G or 24G memory) + CUDA cuDNN

## Getting Started
### Installation
- Install Tensorflow and dependencies from https://www.tensorflow.org/install/
- Install Keras libraries [Keras](https://github.com/keras-team/keras).
```bash
sudo pip install keras
```
- Clone this repo:
```bash
git clone https://github.com/ImagingLab/Colorizing-with-GANs.git
cd Colorizing-with-GANs
```

### Dataset
- We use the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [Places365](http://places2.csail.mit.edu) datasets. To train a model on the full dataset, please download datasets from official websites.
After downloading, please put it under the `datasets` folder in the same way the sub-directories are provided.

## Method

### Generative Adversarial Network
Both generator and discriminator use CNNs. The generator is trained to minimize the probability that the discriminator makes a correct prediction in generated data, while discriminator is trained to maximize the probability of assigning the correct label. This is presented as a single minimax game problem:
<p align='center'>  
  <img src='img/gan.png' />
</p>
In our model, we have redefined the generator's cost function by maximizing the probability of the discriminator being mistaken, as opposed to minimizing the probability of the discriminator being correct. In addition, the cost function was further modified by adding an L1 based regularizer. This will theoretically preserve the structure of the original images and prevent the generator from assigning arbitrary colors to pixels just to fool the discriminator:
<p align='center'>  
  <img src='img/gan_new.png' />
</p>

### Conditional GAN
In a traditional GAN, the input of the generator is randomly generated noise data z. However, this approach is not applicable to the automatic colorization problem due to the nature of its inputs. The generator must be modified to accept grayscale images as inputs rather than noise. This problem was addressed by using a variant of GAN called [conditional generative adversarial networks](https://arxiv.org/abs/1411.1784). Since no noise is introduced, the input of the generator is treated as zero noise with the grayscale input as a prior:
<p align='center'>  
  <img src='img/con_gan.png' />
</p>
The discriminator gets colored images from both generator and original data along with the grayscale input as the condition and tries to tell which pair contains the true colored image:
<p align='center'>  
  <img src='img/cgan.png' />
</p>

## Networks Architecture
The architecture of generator is the same as  [U-Net](https://arxiv.org/abs/1505.04597):  5 encoding units and 5 decoding units. The contracting path has the typical architecture of a convolutional networks: 3x3 convolution layers, each followed by batch normalization, leaky rectified linear unit (leaky ReLU) activation function and 2x2 max pooling operation with stride 2 for downsampling. The number of channels are doubled after each downsampling step. Each unit in the expansive path consists of an upsampling layer, followed by a 2x2 convolution layer that halves the number of channels, concatenation with the activation map of the mirroring layer in the contracting path, and two 3x3 convolution layers each followed by batch normalization and ReLU activation function. The last layer of the network is a 1x1 convolution which is equivalent to cross-channel parametric pooling layer. The number of channels in the output layer is 3 with L*a*b* color space.
<p align='center'>  
  <img src='img/unet.png' width='600px' height='388px' />
</p>

For discriminator, we use a conventional convolutional neural network classifier architecture: a series of 3x3 convolutional layers followed by max-pooling layer with the number of channels being doubled after each downsampling. All convolution layers are followed by batch normalization, leaky ReLU activation with slope 0.2 and dropout with a dropout rate of 20% to prevent the discriminator from overfitting. After the last layer, a convolution is applied to map to a 1 dimensional output, followed by a sigmoid function to return a probability value of the input being real or fake. 
<p align='center'>  
  <img src='img/discriminator.png' width='510px' height='190px' />
</p>
  
## Citation
If you use this code for your research, please cite our paper <a href="https://arxiv.org/abs/1803.05400">Image Colorization with Generative Adversarial Networks</a>:

```
@article{nazeri2018image,
  title={Image Colorization with Generative Adversarial Networks},
  author={Nazeri, Kamyar and Ng, Eric},
  journal={arXiv preprint arXiv:1803.05400},
  year={2018}
}
```
  
## Places365 Results
Colorization results with Places365. (a) Grayscale. (b) Original Image. (c) Colorized with GAN.
<p align='center'>  
  <img src='img/places365.png' />
</p>
