#**German Traffic Sign Recognition** 

##Convolutional Neural Network with TensorFlow

###Join me on this exciting journey to build, train and validate a new deep neural network to recognize German traffic signs! Thanks to Udacity Self-driving Car Nanodegree for provide me the basic skills set to get there!

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./my_images/sample_rgb.png "Dataset Sample"
[image2]: ./my_images/count_each_sign.png "Dataset Distribution"
[image3]: ./my_images/count_each_sign2.png "Augmented Dataset Distribution"

[image13]: ./my_images/shuffle_correct.png "Properly shuffled Images"
[image14]: ./my_images/my_model_AllCNN.jpg "Model Architecture"

[image16]: ./my_images/inception.jpg "Inception Module"
[image17]: ./my_images/wrong_predictions.png "Wrong Predictions"

[image18]: ./my_images/visual_original.png "Original Image"

[image20]: ./my_images/visual_firstlayer.png "1st Layer"

[image23]: ./my_images/visual_secondlayer.png "2nd Layer"

[image25]: ./my_images/custom_images.png "Custom Images from Web"

[image27]: ./my_images/custom_images_performance.png "Custom Images Performance"
[image28]: ./my_images/custom1.png "Top 5 Softmax Image 1"
[image29]: ./my_images/custom2.png "Top 5 Softmax Image 2"
[image30]: ./my_images/custom3.png "Top 5 Softmax Image 3"
[image31]: ./my_images/custom4.png "Top 5 Softmax Image 4"
[image32]: ./my_images/custom5.png "Top 5 Softmax Image 5"
[image33]: ./my_images/custom6.png "Top 5 Softmax Image 6"


####Here is the link to my [Project Code](https://github.com/rzuccolo/rz-carnd-traffic-sign/blob/master/traffic-sign-classification.ipynb)

***
##Data Set Summary & Exploration

####1. The data is available here [Download dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip). First thing, I have loaded the data!

Where in my code:
* Step 0: Load The Data

####2. Next, I got to know the data set.  So first I went into the [source](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) to check from where the data was coming from. Next I played a bit with len(), numpy.shape and print methods to get familiar with the tensor dimensions and the information it was holding.

Where in my code:
* Step 0: Load The Data
* Step 1: Dataset Summary & Exploration

| Parameter         		| Description	        						| 
|---------------------------|-----------------------------------------------| 
| Image shape               | 32x32x3 RGB image   							| 
| Training set length     	| 34799 samples 	                            |
| Validation set length 	| 4410 samples									|
| Test set length 	      	| 12630 samples									|
| Number of classes	      	| 43 classes									|


####3. Next, I have used Matplotlib to visualize a random sample of the training set. 

Where in my code:
* Step 1: Dataset Summary & Exploration

Here is a training data set sample generated with my code:

![alt text][image1]

####4. Next, I have used Matplotlib to plot the histogram of the training labels. It allowed me to visualize the distribution of the training set according to the 43 classes. I noted some classes are much smaller than others, i.e. the data set is unbalanced and could drive my model to be biased if not properly modeled.

Where in my code:
* Step 1: Dataset Summary & Exploration

Here is the training data set histogram per class:

![alt text][image2]

***

##Design and Test a Model Architecture

####1.  Next, I checked the sequence/order of the data set by class and the mean and std values. The sequence results  showed the data need to be shuffled. The mean of about 82 suggested the data need to be normalized. 
####Regarding the normalization I have first tried to subtract and divide by 128 to keep it on an interval of [-1,1]. Here is the code which is not in the final solution file:
```
X_train = (X_train  - 128.) / 128.
X_validation = (X_validation  - 128.) / 128.
X_test = (X_test  - 128.) / 128.
```

####I have also tried later to zero center the data as recommended by [Stanford Winter Quarter 2016 class (CS231n: Convolutional Neural Networks for Visual Recognition. Lecture 5)](https://www.youtube.com/watch?v=gYpoJMlgyXA&list=PLFznuEIsFrh7j2ARuJzbDRb5iZip3rlNR&index=10&t=4003s). Here is the code which is not in the final solution file:

```
X_train = (X_train - np.mean(X_train))
X_validation = (X_validation - np.mean(X_validation))
X_test = (X_test - np.mean(X_test))
```

####But in the end I got better results dividing by 255, to keep it on an interval of [0,1].

Where in my code:
* Step 2: Design and Test a Model Architecture


####2. Unbalanced data set. I have used augmented data. On my limited free time I could not spent much time better understanding yet the opencv functions used to generate the augmented data, yet, so I have mostly directly applied it. Thanks for the functions made available by [Vivek Yadav](https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3#.t2tjvnix2).  I have balanced the data to 1800 samples for each class.

Here is the augmented training set summary. 

| Parameter         	        	| Description	        						| 
|-----------------------------------|-----------------------------------------------| 
| Image shape                       | 32x32x3 RGB image   					| 
| Augmented Training set length   	| 78060 samples (about 124% increase)           |

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Pre-process the Data Set (jittering, normalization, etc.)

Here is the augmented and balanced training data set histogram per class:

![alt text][image3]

####3. Next, the data was shuffled. Then I check the sequence again. All good!

 Where in my code:
* Step 2: Design and Test a Model Architecture

Here is the sequence of shuffled data set:

![alt text][image13]

***
###Model Architecture

####1.  I have started with traditional architectures, alternating convolution and max-pooling layers followed by a small number of fully connected layers, but end up with all conv net model inspired by [Jost Tobias Springenberg, Alexey Dosovitskiy, Thomas Brox, Martin Riedmiller](https://arxiv.org/pdf/1412.6806.pdf)

####This model does not use fully connected layers before the classifier, it uses 1x1 convolutions instead:
>*"...This leads to predictions of object classes at different positions which can then simply be averaged over the whole image. This scheme was first described by Lin et al. (2014) and further regularizes the network as the one by one convolution has much less parameters than a fully connected layer..."*

####Methods I have tried includes:

* RGB 3 channels vs Grayscale 1 channel
* Normalization (128 vs zero-centered vs 255)
* Batch normalization
* Original vs augmented data set
* L2 regularization
* Learning rate decay
* Inception modules
* Spatial transformers 

####It was an intensive learning from all those applications. I am still on the surface of some of those methods like inceptions and spatial transformers but it was good enough to get the intuition and get my feet wet.

Here is the model:

![alt text][image14]

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Model Architecture

####2. Now, a couple of things I have tried but it is not in my final solution.

####On a previews architecture I have tried spatial transformers. Why the spatial transformers (ST)? I am still improving my knowledge on that, but here some very interesting references and insights:
* Spatial Transformer Networks, Google DeepMind, London, UK
[Max Jaderberg / Karen Simonyan / Andrew Zisserman / Koray Kavukcuoglu, arxiv.org publication](https://arxiv.org/pdf/1506.02025.pdf)
[Google DeepMind Presentation, YouTube Video from Xavi Giró-i-Nieto](https://www.youtube.com/watch?v=6NOQC_fl1hQ)
[Symposium: Deep Learning - Max Jaderberg, YouTube Video from Microsoft Research](https://www.youtube.com/watch?v=T5k0GnBmZVI)
>*"...We show that the use of spatial transformers results in models which learn invariance to translation, scale, rotation and more generic warping, resulting in state-of-the-art performance on several benchmarks, and for a number of classes of transformations..."*

* Traffic Sign Classification Using Deep Inception Based Convolutional Networks
[Mrinal Haloi, arxiv.org publication](https://arxiv.org/pdf/1511.02992.pdf)
>*"...obviated the use of handcrafted data augmentation such as translation, rotation etc. and allows the network to learn active transformation of features map..."*

* Better understanding what is behind ST:
[Kevin Zakka's Blog, Github publication](https://kevinzakka.github.io/2017/01/18/stn-part2/)
[Kevin Nguyen, Medium publication](https://medium.com/wonks-this-way/spatial-transformer-networks-with-tensorflow-2eaed0374a16#.scypyqnj0)
>*"...ST allows neural networks to negotiate on its own terms how much it needs to be spatially invariant to the input data..."*
>
>*"...ST’s has potential to replace image pre-processing tasks. Reducing the need for handcrafted features, in turn, leads to better end-to-end deep learning architectures..."*


####Before I have also created and applied an inception module to capture local and global features together.  Maybe with the right model architecture and hyper-parameters a better performance could be achieved with inception modules, as [Mrinal Haloi](https://arxiv.org/pdf/1511.02992.pdf) did, I had not yet tried to replicate this model, may be later! So far I am happy with the adopted methodology and results in my learning path.
Here is the inception module implemented:

![alt text][image16]


 Here are the function, which is not in the final solution file:

```
# Inception module
def inception(x, incNodes=20):
    '''
    This is an inception module, meant to be applied anywhere in the model.
    x = 4D array containing raw pixel data of the images, (num examples, width, height, channels)
    incNodes = number of nodes or output depth for each of the 1x1, 3x3 and 5x5 to be concatenated in the and of the inception. Final output will have 3xincNodes output depth
    '''
    x_shape = x.get_shape().as_list()
    mu = 0
    sigma = 0.1
    weights = {
        'ic_wc1': tf.Variable(tf.truncated_normal(shape=(1,1,x_shape[3],incNodes), mean = mu, stddev = sigma), name='ic_wc1'),
        'ic_wc2': tf.Variable(tf.truncated_normal(shape=(3,3,incNodes,incNodes), mean = mu, stddev = sigma), name='ic_wc2'),
        'ic_wc3': tf.Variable(tf.truncated_normal(shape=(5,5,incNodes,incNodes), mean = mu, stddev = sigma), name='ic_wc3')
    }

    biases = {
        'ic_bc1': tf.Variable(tf.zeros(incNodes), name='ic_bc1'),
        'ic_bc2': tf.Variable(tf.zeros(incNodes), name='ic_bc2'),
        'ic_bc3': tf.Variable(tf.zeros(incNodes), name='ic_bc3')
    }

    # Average Pooling 3x3.
    avgpol1 = avgpool2d(x, k=3, S=1, padding='SAME') 
    #print("avgpol1 shape:",avgpol1.get_shape())

    # Convolution 1x1.
    c1x1avgpool = conv2d(avgpol1, weights['ic_wc1'], biases['ic_bc1'], padding='SAME') 
    #print("c1x1avgpool shape:",c1x1avgpool.get_shape())

    c1x1 = conv2d(x, weights['ic_wc1'], biases['ic_bc1'], padding='SAME') 
    #print("c1x1 shape:",c1x1.get_shape())

    # Convolution 3x3.
    c3x3 = conv2d(c1x1, weights['ic_wc2'], biases['ic_bc2'], padding='SAME') 
    #print("c3x3 shape:",c3x3.get_shape())

    # Convolution 5x5.
    c5x5 = conv2d(c1x1, weights['ic_wc3'], biases['ic_bc3'], padding='SAME') 
    #print("c5x5 shape:",c5x5.get_shape())

    # Concatenate
    conc1 = tf.concat(3, [c1x1,c3x3,c5x5])
    #print("conc1 shape:",conc1.get_shape())

    return conc1
```

####Batch Normalization.  Amazing blog I found, [R2RT](http://r2rt.com/implementing-batch-normalization-in-tensorflow.html), need spend more time there. Tried to find the author name in the blog, could not.
>*"...Batch normalization, as described in the March 2015 paper (the BN2015 paper) by Sergey Ioffe and Christian Szegedy, is a simple and effective way to improve the performance of a neural network. In the BN2015 paper, Ioffe and Szegedy show that batch normalization enables the use of higher learning rates, acts as a regularizer and can speed up training by 14 times. In this post, I show how to implement batch normalization in Tensorflow..."*

####Again, in the end I got better results without it. Here are my functions, which is not in the final solution file:

```
# Convolution batch normalization helper
def conv2dBatchN(x, W, b, output_depth, strides=1, padding='VALID'):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
                     padding=padding)
    
    # Batch Normalization
    # http://r2rt.com/implementing-batch-normalization-in-tensorflow.html  
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x,[0])
    scale = tf.Variable(tf.ones([output_depth]))
    beta = tf.Variable(tf.zeros([output_depth]))
    x = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,scale,epsilon)
    print('x shape:', x.get_shape())
    
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


# Fully connected batch normalization helper
def fc_reluBatchN(x, W, b, output_depth):
    x = tf.add(tf.matmul(x, W), b)
    
    # Batch Normalization
    # http://r2rt.com/implementing-batch-normalization-in-tensorflow.html  
    epsilon = 1e-3
    batch_mean, batch_var = tf.nn.moments(x,[0])
    scale = tf.Variable(tf.ones([output_depth]))
    beta = tf.Variable(tf.zeros([output_depth]))
    x = tf.nn.batch_normalization(x,batch_mean,batch_var,beta,scale,epsilon)
    
    return tf.nn.relu(x)
```


###Train, Validate and Test the Model

####1. Same as  [Muddassir Ahmed](https://medium.com/@muddassirahmed/german-traffic-sign-classification-using-deep-learning-219c53fba329#.64ca6z3i3), I have used L2 regularization and Rate of learning rate decay.

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Train, Validate and Test the Model

####2. Cross Entropy, Softmax, AdamOptimizer and Evaluate accuracy function are same implementation as in the LeNet nanodegree lesson.

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Train, Validate and Test the Model

####3. I have over fitted a small sample of the training to evaluate the model efficiency as recommended by [CS231n Winter 2016: Lecture 5: Neural Networks Part 2](https://www.youtube.com/watch?v=gYpoJMlgyXA&index=10&list=PLFznuEIsFrh7j2ARuJzbDRb5iZip3rlNR&t=2059s) and also implemented by [Muddassir Ahmed](https://medium.com/@muddassirahmed/german-traffic-sign-classification-using-deep-learning-219c53fba329#.64ca6z3i3). 20 images sample has been over fitted.

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Train, Validate and Test the Model

####4.  Here my final choice for Hyper-parameters:
* EPOCHS = 60
BATCH_SIZE = 512

* Initial learning rate
lr = 0.00095

* L2 regularization (Beta)
b = 1e-6

* Rate of learning rate decay
k = 1e-5


Where in my code:
* Step 2: Design and Test a Model Architecture
* *Train, Validate and Test the Model

####5. Final model has been evaluated with an accuracy of 98%!!

 Where in my code:
* Step 2: Design and Test a Model Architecture
* *Train, Validate and Test the Model

Here is a sample of the wrong predictions:

![alt text][image17]


####6. Finally I was interested in visualize how my model "see" the images throughout the layers. Thanks to [Arthur Juliani](https://medium.com/@awjuliani/visualizing-neural-network-layer-activation-tensorflow-tutorial-d45f8bf7bbc4#.h0mazchg3)  for some code instructions. Check below the 1st and 2nd layers for a random sample image of the training set: 

Original image:

![alt text][image18]

1st layer:

![alt text][image20]

2nd layer:

![alt text][image23]


***
###Test a Model on New Images

####1. I got six (6) German traffic signs from the web. I tried to get sign with busy background landscape and with some perspective distortion. The bumpy road sign (5th) seems a bit challenge due to low brightness, reflection in the background and the perspective distortion.

 Where in my code:
* Step 3: Test a Model on New Images
* *Load and Output the Images

Here are the six (6) German traffic signs that I found on the web:

![alt text][image25]

####2. The predictions were evaluated. The accuracy on the captured images is 100% while it was 98% on the testing set, six images from the web is a small sample to be compared to a comprehensive test set of 12630, but we can say it works good!

Where in my code:
* Step 3: Test a Model on New Images
* *Predict the Sign Type for Each Image
* *Analyze Performance

Here is the analyze performance visualization:

![alt text][image27]


####3. Next, I will show how certain the model is when predicting on each of the six (6) new images by looking at the softmax probabilities for each prediction. Top five (5) softmax probabilities for each image along with the sign type of each probability is presented. 

Where in my code:
* Step 3: Test a Model on New Images
* *Analyze Performance

The top five (5) softmax probabilities for each image:

![alt text][image28]
![alt text][image29]
![alt text][image30]
![alt text][image31]
![alt text][image32]
![alt text][image33]
