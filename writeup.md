##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_noncar.jpg
[image2]: ./output_images/hog_example.jpg
[image3]: ./output_images/hog_noncar_example.jpg
[image4]: ./output_images/find_car.jpg
[image5]: ./output_images/heatmap.jpg
[video1]: ./detect_cars.mp4
[video2]: ./test_video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in code cell #400 of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `RGB` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` for both a car and a non-vehicle:

![alt text][image2]
![alt text][image3]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters (each item in list is a choice):
orientation = [7,8,9]
pixels_per_cell = [(8,8),(16,16)]
hog_channel = [0,'ALL']
spatial_size = [(16,16),(32,32)]
hist_bins = [32,16]

Eventually, I found that the following combination has the best results:
orientation = 7
pixels_per_cell = (8,8)
hog_channel = 'ALL'
spatial_size = (32,32)
hist_bins = 32

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In code cell 334, I trained a linear SVM using a stack of hog features, spatial binning and color histograms.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This portion of the code can be found in cell 436. 
The cars are most likely going to occur on the road, which is below 400 pixels vertically, so I set ystart to 400. I also tried different scale of windows. Since I trained the model on 8x8 cells, I kept the window size the same. Then I proceeded to trying scales from 0.8 all the way to 2. I realized scales between 0.8 - 1.5 work the best with a step size of 2 pixels (instead of defining overlap)

![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Data selection is extremely important because machine learning follows garbage in garbage out. At first, I loaded in all images of cars and non-cars I have and randomly shuffled them before splitting to train and test. What I found was that no matter which classifier I choose, accuracies were extremely high. So, I decided to get most of my training data from 5 folders and test data from the rest 2 folders. Since images in each folder are likely to be similar to each other, this way my model is tested on images it has never seen before. 


I tried many classifiers including logistic regression, adaboost, gradient boosting and such. But I realized Linear SVC actually works the best with a test accuracy of 0.84. All others were below 0.8.

I tried almost all the color spaces available: RGB, LUV, HLS, HSV and etc. Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  The first scale of 1.5 starts at the top of the road (ystart = 400) and end above the car (ystop = 656). The second scale of 1.2 focuses on detecting smaller vehicles that appear far away on the road, so it has ystart=400 and ystop=500. I also set a threshold of 2 for the heatmap. Here are some example images:

![alt text][image5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./detect_cars.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.
 
I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I then created a vehicle class to keep track of the detected objects after heatmap. If the objects in the current detection overlaps with any of the existing vehicles, it is marked as the same vehicle (code cell 365). Only when a vehicle has appeared over five times in the past 10 frames, a box is drawn to recognize it. It establishes a fairly high threshold for objects to be detected, thus filtering out the false positives.

Here's an example result showing the pipeline working on the test video:
[link to my video result](./test_video_out.mp4)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I was stuck on a very basic problem for a long time. I had scaled my training and test data but when training the classifier, I used the original X_train instead of the scaled version. My video performance had a lot of false positives because of that. This shows how important it is to examine the code line by line especially when copying the code from somewhere else.

I also tried different ways of smoothing. The first one was to store heatmaps from different frames and overlay them. The second one is to track vehicles detected using a Vehicle class. Lastly, I tried using openCV's groupRectangles class in an attemp to group smaller boxes. Eventually, I went with the second one, because it helped me smooth the frame and keep track of different properties of the objects detected. It also filtered out false positives, and filled in the gap when the vehicle was not detected in one specific frame.

If I had more time to work on this project, I would try more parameters in HOG transformation and window searching. Currently I have 3 window searches for each frame but I imagine increasing that and use corresponding horizons would be helpful.

