## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.


The Project
---

The goals / steps of this project are the following:

* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image2]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/README_images/mean_variance.png "Well-Conditioned"
[image4]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/german_signs/pic1.png "Traffic Sign 1"
[image5]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/german_signs/pic2.png "Traffic Sign 2"
[image6]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/german_signs/pic3.png "Traffic Sign 3"
[image7]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/german_signs/pic4.png "Traffic Sign 4"
[image8]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/german_signs/pic5.png "Traffic Sign 5"
[image9]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/README_images/softmax_probability.png "Softmax Probability"
[image10]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/README_images/training_dist.PNG "Training Distribution"
[image11]: https://github.com/white315/CarND-Traffic-Sign-Classifier-Project/blob/master/README_images/validation_dist.PNG "Validation Distribution"

### Notes Concerning the Dataset

Instructions:
1. Download the data set. The classroom has a link to the data set in the "Project Instructions" content. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

Response:
I decided to slice a validation set from the training set as opposed to using the already available validation set. This for two reasons: 1) More coding practice, 2) various classmates have used the same approach, trying to get a more realistic sense of the industry's practices.

## [Rubric](https://review.udacity.com/#!/rubrics/481/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Dataset Exploration

#### 1. Dataset Summary

I loaded both the training set and the validation set from the workspace's [data](https://github.com/white315/CarND-Traffic Sign Classifier-Project/blob/master/data) folder. By doing an initial run through with the LeNet architecture, I achieved a little over 80% Validation Accuracy (initially, I had used the validation set from the data folder).

#### 2. Exploratory Visualization

Just like in the LeNet Lab, I randomly selected a sign image and compared it to the label to ensure I was gathering images and labels correctly in the third code cell.

I wanted to get a sense of how many samples I had to work with. Originally, I performed histograms on the labels for the training, validation, and test dataset. Eventually came to the conclusion that only the training set mattered to me since I would by slicing a validation set anyway and the test set is supposed to be blind to the computer anyway - why not also be blind to me?

This visualization was also performed to check that the augmentation was done correctly (at minimum, 1,000 samples of each label are created) and after slicing a validation set to compare variance. I reapplied a title and axis labels and so lost the 1,000 sample minimum's original histogram.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

First, I converted to grayscale. This was done for two reasons: 1) to reduce test time since initial depth is lower and 2) darker images can get better visibility. Number 2 was proven by [Neta Zmora](https://github.com/netaz) in her repository.

Then, I normalized the image data to try to get to a "well-conditioned" state as discussed in lecture video (Lesson 10, Slide 25. Normalized Inputs and Initial Weights) and visualized below:

![Well-Conditioned][image2]

After doing some research on sign sample quantity, I decided to add more images by randomly augmenting training images for lower quantity samples. I had worked with a minimum of 400, 450, 600, and finally 1,000 minimum samples per label to achieve higher and higher accuracy. Eventually, going from 92% with the 400 minimum samples to 99% with the 1,000 samples.

I performed this randomization in the sixth code cell by chaining modifications like translating (image shift using 'translate()', scaling (just like image modification on roads using 'scale()' warping (twisting or rotating using 'warp()'), and randomizing brightness (using 'brighten()').

This data was saved in the fourteenth code cell so that I could reload the augmented data. The augmented data was stored in the [augmented_data](https://github.com/white315/CarND-Traffic Sign Classifier-Project/blob/master/augmented_data) folder location. When it gets loaded, it gets renamed into pipeline-friendly names.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model is located in the ninth code cell and consists of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,   outputs 16x16x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,   outputs 5x5x16  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x400 	|
| RELU					|												|
| Flatten				| Input of 1x1x400, output of 400				|
| Flatten				| Input of 5x5x16, output of 400				|
| Concatenate			| Input both flattened to give output 800		|
| Dropout				| 												|
| Fully connected		| Inputs the 800 labels and outputs 120 labels	|
| RELU					|												|
| Fully connected		| Inputs the 120 labels and outputs 43 signs	|
 
This is similar to the LeNet structure and the changes are pointed out in lines: 49*, 54*, 57**, 63**, and 66***. I also removed the third fully connected layer.

49 & 54 are a third convolution to try out the 1x1 convolution idea
57 & 63 are using the third convolution to flatten and concatenate with another flattened layer
66 is dropout practice

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I...
1. Used the model architecture to gather the logits
2. Performed softmax cross entropy to compare the logits with "ground truth" or actual labels
3. Averaged the cross entropy across the training dataset
4. Used the AdamOptimizer to minimize the loss function
5. Performed backprop to minimize training loss using the 'optimizer.minimize()' function

Over time, I used a batch size and learning rate that came up a lot during labs and quizzes as an ideal choice: 100 and 0.001 respectively. The number of Epochs went from 10 to 20 to 50 and finally landed on 30, only because I didn't want to rerun the pipeline to optimize. I found that between Epoch 17 and 30, there was no meaningful difference between accuracy, so leaving the Epoch value at 20 would have been adequate and still have yeilded a 99% validation accuracy.

Final hyperparameters:
```python
EPOCH = 30
BATCH_SIZE = 100
learning_rate = 0.001
mu = 0
sigma = 0.1
```

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The first structure used was an exact replica of the LeNet Lab sandbox while also using images that were unprocessed. This resulted in a validation accuracy of 85%
The second iteration included preprocessing (grayscale and normalization. This resulted in validation accuracy of 87%
After the second iteration, I started changing up the Model Architecture, finally getting it to where it is now, which only resulted in 91-92% validation accuracy.
I went to tuning the hyperparameters but found that the default learning rate's change didn't help all that much, and sigma changes were too drastic to be meaningful to me without correlated visualization. I finally was able to change batch size and get a meaningful result, getting consistently above 92% at 10, 20, and 50 Epochs.
This is when I dropped out the validation set and started augmenting images. I had landed on 30 Epochs by this time. By splicing the validation set and increasing the training sample sizes to minimums, I yielded the following:

| Minimum         		|     Validation Accuracy      		| 
|:---------------------:|:---------------------------------:| 
| 400					| 92%								|
| 450					| 93%								|
| 600					| 97%								|
| 1,000					| 99%								|

Final accuracy occurred when the training set looked like this:

![Training Distribution][image10]

And the validation set looked like this:

![Validation Distribution][image11]

My final model results were:
* Validation Accuracy: 99.2%
* Test Accuracy: 93.8%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Keep Right][image4] ![60 km/h][image5] ![Left Turn][image6] 
![30 km/h][image7] ![Road Work][image8]

The first and third image might be difficult to classify because arrows are common for German road signs. Lucking the arrowheads themselves are quite different so "Keep" and "Turn" can be distinguished from eachother.

The second and fourth sign both have numbers and, as can be seen when looking at softmax probabilities, similar images involve numbers.

The fifth image has low resolution and German road signs often use the red border triangle to communicate information. This could be confused with another informational road sign.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        			| 
|:---------------------:|:-------------------------------------:| 
| Keep Right      		| Keep Right   							| 
| 60 km/h     			| 60 km/h 								|
| Left Turn				| Left Turn								|
| 30 km/h	      		| 30 km/h   					 		|
| Road Work 			| Road Work         					|


The model was able to correctly guess all 5 of the traffic signs correctly, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.8%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 23rd cell of the Ipython notebook.

Seemingly, there is a high confidence that the signs are labeled correctly as all best guesses are 100% correct (see below):


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?




