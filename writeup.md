#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Image References

Dataset used for training, validating, and testing: [traffic-signs-data.zip](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip).

Follow along in the notebook [here](report.html#Step-0:-Load-The-Data).

## Rubric Points

###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---

###Writeup / README

####1. The submission includes the following:

- This writeup
- [notebook](Traffic_Sign_Classifier.ipynb) and its [html](report.html) version
- The whole [project](https://github.com/ysono/CarND-T1P2-Traffic-Sign-Classifier-Project/) and its [readme](README.md)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Follow along in the notebook [here](report.html#Provide-a-Basic-Summary-of-the-Data-Set-Using-Python,-Numpy-and/or-Pandas).

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Follow along in the notebook [here](report.html#Include-an-exploratory-visualization-of-the-dataset).

First, a histogram is plotted to show the uneven distribution of classes in the training dataset.

Then, a sample of each class is shown by choosing the first instance of that class in the training dataset (without shuffling the dataset).

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The decisions on pre-processing and generation operations were iterative: for each change, a model was trained on the changed dataset and its accuracy was evaluated. These decisions were retained on rejected based on improvements on the model and performance side effects.

Therefore, the presentation in this writeup and in the notebook is not chronological. The effect of each operation is visualized by applying that operation alone on the raw dataset. Follow along in the notebook [here](report.html#Pre-process-the-Data-Set-(normalization,-grayscale,-etc.)).

##### 1. Grayscale

Training on grayscale images improved validation accuracy. It is hypothesized that the color information was an overall distractor because too much variation existed in the training dataset, due to natural conditions (perhaps human eyes are too adept at normalizing colors and this leads to overestimation of the importance of colors).

Later, when generating images by adding gaussian noise, it was found that generating random values for only 1 grayscale layer vs 3 rgb layers vastly improved performance, and this became another reason for retaining grayscale.

##### 2. Normalization of pixels around zero

This did not affect model performance or accuracy noticeably, but it should help prevent gradient instability.

In order to normalize to the correct continuous range of scalars, the image was first converted from unsigned int to float.

##### 3. Generating images by cropping

This operation should improve translation invariance of the model.

Cropping at the boundary of the sign as identified in the raw dataset introduces occlusion as well, because the boundaries are often too small and are cutting into the perimeter of the sign.

##### 4. Generating images by distorting

Images are distorted by vertically and horizontally compressing the originals and filling the periphery with zero-valued pixels. The paddings are calculated so as to never compress height or width to zero, but since the paddings are chosen randomly, they can occupy the majority of the images.

A more robust transformation would also use opencv's affine transformations to rotate images or stretch them diagonally.

##### 5. Generating images by adding gaussian noise

Noise was generated using `np.random.normal()`. The standard deviation of 0.04 was chosen by visual inspection to have noticeable but not excessive blotches.

Before noise was added, images must first undergo the normalization above which constrains pixel values to `[1, -1]`; if the order were to be reversed, the effect of noise would be reduced by a factor of 128.

At one point, training with dataset that had noise added (which doubled the size of the dataset) increased the validation accuracy by 2 percentage points.

##### 6. Pipeline

For every original raw training image:

1. Generate 2 additional images, by cropping and by distorting.
1. Pre-process: grayscale then normalize.
1. Add noise. This duplicates the amount of images in the pipeline, for a total of 4 per original.

An alternative was:

1. Geneerate 3 additional images, by cropping, by distorting, and by adding noise
1. Pre-process: grayscale then normalize.

This alternative would yield 3x images per original. However, as mentioned above, generating random values for 3 rgb layers (only to later have them merged into 1 grayscale layer) was much more expensive. Generating 4x was cheaper than 3x. For this reason, the alternative pipeline was rejected.

##### 7. Evenly distribute classes in the training dataset

After the multiplicity of the generation pipeline was determined, the amount of generation required was calculated across each class, such that all classes would end up with about the same amount of samples.

As an undesirable side effect, the size of the training dataset became completely skewed w.r.t. the validation dataset. Whereas the original ratio of validation vs trainig was `4410/34799 = 12.7%`, after generation, it became `4410/464502 = 0.9%`. This could explain the slow improvement in training accuracy w.r.t. validation accuracy, to be seen below. As a possible remedy for this problem, multiplicity of generation should be reduced for populous classes, e.g. by applying the pipeline to a portion of the raw dataset that is inversely proportional to the representation of the class. Another remedy could be to reallocate some samples from raw training dataset to validation; if we were aiming for `70/15/15` distribution, `15/70 = 21.4%`, and the original 12.7% was already too low.

##### 8. Summary

In LeCun's paper (Sermanet et al., 2011), he mentions

> real-world variabilities such as viewpoint variations, lighting conditions (saturations, low-contrast), motion-blur, occlusions, sun glare, physical damage, colors fading, graffiti, stickers and an input resolution as low as 15x15

The generation pipeline attempts to replicate some of these natural effects and apply them onto the raw dataset that already have these effects.

The goal of pre-processing is to constrain the scalar of each pixel to `[-1, 1]`. The combination of grayscaling and normalizing changes the per-pixel memory from `3 * uint8 = 3 bytes` to `1 * float32 = 4 bytes`.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer                 | Description                                   |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x1 grayscale image                       |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x6    |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x16     |
| Fully connected       | from 5x5x16=400 neurons to 120 neurons        |
| RELU                  |                                               |
| Dropout               | 50%                                           |
| Fully connected       | from 120 to 84 neurons                        |
| RELU                  |                                               |
| Fully connected       | from 84 to 43 neurons                         |

These aspects of the model were retained from the CarND LeNet lab:

- The sizes of the convolution and fully-connected layers. They already worked well, and changing them did not yield any improvements.
- Strides were kept at 1 to prevent excessive loss of information on inputs that were already low in resolution.
- The use of RELU, in both convolution and fully-connected layers, due to emperical evidence of its superiority over sigmoid.
- The randomnization of weights (`mu = 0; sigma = 0.1`). These were sensitive hyperparameters which greatly affected the rate of improvement of validation accuracy during early epochs.

See [here](https://github.com/udacity/CarND-LeNet-Lab) for a diagram of LeNet architecture.

Dropout was added. Its location was chosen by experiment to be at the layer with 120 outputs, and `keep_prob` was chosen at 50%.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The output of the model was interpreted as softmax logits, and cross-entropy was chosen as the cost. L2 cost was added for regularization. The L2 multiplier (lambda / n) was initially chosen at 0.1, and then divided by 10 till the optimal value of 0.0001 was found.

The batch size of 10 was retained from the CarND LeNet lab, with the assumption it was optimized for aws g2.2xlarge.

Consistently, the validation accuracy stayed higher than the training accuracy over all epochs, regardless of choice of hyperparameters. Both accuracies were generally improving, suggesting there was no overfitting yet. The two accuracies were also converging, as the improvement in the validation accuracy was slower than the improvement in the training accuracy. This was happening before L2 and dropout regularizations were added, and before massive data generation was implemented that skewed the relative size of the validation dataset to 0.9%.

This skewing did, however, slow down the improvement of the training accuracy. The learning rate was increased from 0.001 to 0.005, but this caused validation accuracy to fall from 95% to 92%, so 0.001 was retained. Instead, the number of epoch was increased to 15. Learning rate and number of epochs was adjusted last.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 93.3%
* validation set accuracy of 95.1% 
* test set accuracy of 93.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  - The architecture was based on LeNet, because it has been successfully trained on traffic signs.
* What were some problems with the initial architecture?
  - Although it is not clear whether the original LeNet has inherent problems, the initial architecture was severely underperforming, at sub-10% validation accuracy after 10 epochs, until the correct weight initialization (mu, sigma) and learning rate were chosen.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  - Adding more convolutional layer may help if traffic signs allowed different positional composition of the same features. However, traffic signs are fairly rigid.
  - Adding more depth to existing convolutional layers may help if there is a need to recognize a larger variety of features, such as lines curved in a specific way. But LeNet's configuration seems to be sufficient.
* Which parameters were tuned? How were they adjusted and why?
  - Sections above describe how hyperparameters were tuned. Among the sensitive ones were weight initialization, dropout rate, L2 multiplyer, and learning rate.
  - The graph of training and validation accuracy was the sole guidance for tuning (I did not cheat by using test accuracy).
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  - Convolution helps greatly with translation invariance.
  - Dropout is thought to help by reducing reliance of any given neuron on any specific subset of its input neurons.

If a well known architecture was chosen:
* What architecture was chosen?
  - The architecture was based on LeNet
* Why did you believe it would be relevant to the traffic sign application?
  - LeNet was successfully trained on similar traffic signs. Traffic signs have the advantage of being "unique, rigid and intended to be clearly visible ..., and have little variability in appearance" (Sermanet et al., 2011); hence any architecture that is proven to work with approximately 43 classes of such images should also work.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  - The test accuracy and training accuracy are close enough to curtail fears of overfitting, and are both high enough to approximate human capability.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web: [as files](test_images_extra) and [in the notebook](report.html#Load-and-Output-the-Images).

Among the 10 images chosen, 2 that are named `*.tricky.jpg` were thought to be difficult to recognize.

- `18.tricky.jpg` because it contains the desired features in the upper half, and the other half contains distracting features that are not seen in the training dataset.
- `31.tricky.jpg` because it is a valid real-world variation for which there is no adequate trainig. The cow it contains in the triangle is valid for class 31 ("Wild animals crossing"), but most (or all?) training samples for this class contain deer. The blob of the cow legitimately resembles other blobs, such as those of a truck or an arrow. This shows safety issue that would occur if the trained model were to be deployed in the real world.

Additional tricky images could have been found that contained stickers/graffitis, shadows, glares, etc.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Follow along in the notebook [here](report.html#Output-Top-5-Softmax-Probabilities-For-Each-Image-Found-on-the-Web).

As expected the 2 "tricky" images were not classified correctly. All others were, however, resulting in the overall accuracy of `8/10 = 80%`.

A direct comparison of this 80% figure against the test accuracy (94%), validation accuracy (95%), or the training accuracy (93%), would not be meaningful because:

- the additional test sample size is miniscule at 10
- they were purposely handpicked such that 20% were expected to fail due to above reasons and the remaining 80% were chosen with clear features to ensure their success.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Follow along in the notebook at the same location [here](report.html#Output-Top-5-Softmax-Probabilities-For-Each-Image-Found-on-the-Web) and scroll down.

Example of images that yielded ...

A *large* gap between the two 1st choices and *high* confidence on the 1st choice:

- `12`. This probably owes to the diamond shape that is unique among the classes

A *large* gap between the two 1st choices but *low* confidence on the 1st choice:

- `35` and `32`. Both contain features (circular outline, arrowhead, and diagonal cross) that are shared by other classes.

A *small* gap between the 1st choices but *high* confidence on the 1st choice:

- The non-tricky `18`. It is conjectured that the rotation and skewed enabled the non-1st choices to be likely.

A *small* gap between the 1st choices and *low* confidence on the 1st choice:

- `36` was almost a mismatch (13% vs 12% top choices). The watermark could have caused the right-pointing arrow to be considered as a potential noise, leading to the 2nd choice being the straight arrow.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


TODO!!


### References

P. Sermanet and Y. LeCun. Traffic sign recognition with multi-scale convolutional networks. In Proceedings of International Joint Conference on Neural Networks (IJCNN’11), 2011.
