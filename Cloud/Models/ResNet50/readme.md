## ResNet50 Transfer Learning Code


* ResNet-50 model pretrained on ImageNet and imported from the Keras applications. 
* Creates the ResNet-50 model and downloads the weights pretrained on the ImageNet dataset.
* ResNet-50 expects the images to be 224 x 224 pixels in size so we used the tf.image.resize() function to resize our images
* ImageDataGenerator to load the images and augment them in various ways.
