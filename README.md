# CNN

This is one of my projects at the Deep Learning course where I built my a CNN model on the CIFAR-10 dataset. I started transfer learning and I referred the VGGNet as my starting model architecture (Very Deep Convolutional Networks for Large-Scale Image Recognition, https://arxiv.org/abs/1409.1556). In that paper, it used 4 blocks of convolutional layers ( each includes 2 convolutional layers + 1 max pooling with ReLu activation) and fully connected layers on the 224 * 224 images with 1000 classes. In the CIFAR-10, the images are 32 * 32, indicating that we may need less convolutional layers to reach a good accuracy in our case. Also, I want to select 2 fully connected layers and an output layer with softmax in order to fully connected the complicated features after convolutional layers . In terms of the number of filters in each layer, I chose to keep them same with VGG, which are also common options in CNN architecture. Therefore, I selected 3 blocks of convolutional layers (2 convolutional layers + 1 maxpooling with ReLu activation) and 3 fully connected layers (both with ReLu activation) adding drop-out and Batch norm in the optimization.



