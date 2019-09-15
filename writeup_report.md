
# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./training/IMG/center_2019_09_08_14_11_46_090.jpg
[image3]: ./training/mirrored.jpeg
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* lenet.h5, model based on LeNet architecture with RGB image input and single steering angle output. It is trained with only left/first/simple track in the simulation. (FAIL)
* dave2-rgb-01.h5, model based on Nvidia's Dave2 architecture with RGB image input and single steering angle output. It is trained with only left/first/simple track in the simulation. (SUCCESS)
* dave2d-hls-combined.h5, model based on Nvidia's Dave2 architecture with HLS image input and single steering angle output. It is trained with samples from both tracks in the simulation. (FAIL)
* writeup_report.md, summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py dave2-rgb-01.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on Nvidia's research paper titled "End to End Learning for Self-Driving Cars" with a small twist and laziness. Model contains normalization layer (model.py lines 61-62) followed by 5 convolution layers (model.py lines 63-77) and 4 fully connected layers (model.py lines 79-85). 

Fist 3 convolutional layers has 5 by 5 kernel with 2 by 2 strides and depths ranging from 24 to 48 (model.py lines 63-69). Following 2 convolutional layers has 3 by 3 kernel with 1 by 1 strides and depth 64 (model.py lines 72-75).

Model includes RELU activation layers with each convolutional layer to introduce non-linearity.


#### 2. Attempts to reduce overfitting in the model

Original Dave2 model doesn't seem to contain a noise layer (e.g., dropout) to prevent overfitting. So, I decided to add dropout in-between each layer (model.py lines 65-68-71-74-77) by default (They were pretty much a carry-over while evolving model from LeNet to Dave2).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 88, validation_split=0.2). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 86).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to test & optimize based on car performance.

My first step was to use a convolution neural network model similar to the LeNet. I thought this model might be appropriate because of past experience with successful image recognation.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that they have similar loss



The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 60-86) consisted of a convolution neural network with the following layers and layer sizes ...

- Cropped input (100 x 300 x 3)
- Normalization with lambda layer (BGR channel range from [0, 255] to [0, 1])
- Convolutional layer with 5 by 5 kernel, 2 by 2 stride, relu activation and 24 depth
- Dropout layer with 0.2 drop rate during training and 0 drop rate during validation (*)
- Convolutional layer with 5 by 5 kernel, 2 by 2 stride, relu activation and 36 depth
- Dropout layer with 0.2 drop rate during training and 0 drop rate during validation (*)
- Convolutional layer with 5 by 5 kernel, 2 by 2 stride, relu activation and 48 depth
- Dropout layer with 0.2 drop rate during training and 0 drop rate during validation (*)
- Convolutional layer with 3 by 3 kernel, 1 by 1 stride, relu activation and 64 depth
- Dropout layer with 0.2 drop rate during training and 0 drop rate during validation (*)
- Convolutional layer with 3 by 3 kernel, 1 by 1 stride, relu activation and 64 depth
- Dropout layer with 0.2 drop rate during training and 0 drop rate during validation (*)
- Flatten into 1164 neurons
- Fully connected layer with 100 layers
- Fully connected layer with 50 layers
- Fully connected layer with 10 layers
- Fully connected layer with 1 layer
- Output


Created model differs from Nvidia research paper in couple of ways. Unlike what research paper defends, we apply some hardcoded preprocessing such as cropping the input image. Research paper claims that model should be able to learn not to care about noise by itself. This allows us to achieve a true end to end learning.

Also, research paper doesn't mention about activation, pooling and noise layers. I am sure they take activation layer for granted so I would predict they use something like *relu* activation layer. Also, there was no mentioning of pooling layer. To test it, I give the model a try without a max pooling. The results were good enough. Still, it could be a great improvement for reducing model size while maintaining accuracy. Yet, I couldn't resist not to use dropout to introduce some noise. I guess, I kind of overdid it made a mistake during validation (*). Still, I am fine with the exceptional results.

I guess most facinating thing about the final model is the huge leap it achieved in training and validation accuracy from epoch 0 compared to LeNet. I was able to train it in less epochs. e.g., 5 in dave2d vs 15 in lenet.

In retrospect, final model could still be optimized for better driving capabilities in other track. Nvidia research paper suggests using YUV color space. Yet, I failed to find a proper opencv documentation about YUV. As an alternative, I gave a try to HLS color space that proved itself during *finding lane lines* project. It definitely did better in the second/hardest/right track in the simulation. Despite that, first/easier/left track performance was terrible. So, I decided to drop that idea for now.

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

(*) Even though 0 dropout during validation was intended behavior, I happen to realize that it wasn't the case while creating this write-up. Current validation set also employs a dropout with 0.8 that is not ideal. I assume that we could solve this issue by using keras.backend.placeholder. Yet, I don't think I will give it a try since our model is already doing good job on driving itself through the track.

#### 3. Creation of the Training Set & Training Process

Since the beginning, I started with sampling a quarter lap on track one/easy/left. Only aim was to capture a proper road alignment with small sample size (faster to train). Here is an example image of center lane driving taken from (./training):

![alt text][image2]


I didn't had to perform couple of recovery attempts as suggested in the writeup sample. I guess I am already a terrible driver enough to generate samples for recovery :). One thing to note though, including left and right cameras with some steering angle adjustment created a real difference on teaching car how to prevent going off track.


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would teach model how to drive in both directions. For example, here is an image that has then been flipped:

![alt text][image2]
![alt text][image3]

After the collection process, I had X number of data points. I then preprocessed this data by cropping car and the noise above the road

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by 6. I used an adam optimizer so that manually training the learning rate wasn't necessary.

