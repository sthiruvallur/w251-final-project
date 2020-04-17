# w251-final-project
Code for the final project


## Edge

To build docker image for Jetson
./Edge/build_docker_image.sh

To run the docker container 
./Edge/run_container.sh

Once inside the container, run the hand capture and predict image by
python3 hand_detect_pb.py

Supplemental information on other options with the container will be found in
./Edge/bashcode

## Cloud

### VGG-16 Transfer Learning
![VGG-16](./Cloud/Models/VGG16/VGG-16_layers.png "VGG-16 Layers")

* VGG-16 so named for its 16 trainable layers ( 13 convolution + 3 dense layers)
* The model was imported Keras applications
* Pre-trained model weights was obtained from F.Chollet's github: [F.Chollet's github](https://github.com/fchollet/deep-learning-models/releases)
* The images from ASL dataset was sized down into 50X50 and trained for 10 epochs
* Notebook and standalone python script is available under Cloud/Models/VGG16/


### ResNet50 Transfer Learning
* ResNet-50 model pretrained on ImageNet and imported from the Keras applications.
* Creates the ResNet-50 model and downloads the weights pretrained on the ImageNet dataset.
* ResNet-50 expects the images to be 224 x 224 pixels in size so we used the tf.image.resize() function to resize our images
* ImageDataGenerator to load the images and augment them in various ways.

![ResNet50](./Cloud/Models/ResNet50/resnet50architecture.png)
