
# Advanced Lane Finding Project

## 1. Project Outlines

The goals and steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undist_calibration1.png "Calibration Chessboard"
[image2]: ./output_images/undist_test_1.png "Undistorted Test Image 1"
[image3]: ./output_images/undist_test_2.png "Undistorted Test Image 2"
[image4]: ./output_images/undist_test_3.png "Undistorted Test Image 3"
[image5]: ./output_images/combined_gray.png "Binary Image of Combined Gradient Thresholds"
[image6]: ./output_images/color_grad_binary.png "Binary Image of Combined Color and Gradient Thresholds"
[image7]: ./output_images/thresh_grad_1.png "Thresholded Image 1"
[image8]: ./output_images/thresh_grad_2.png "Thresholded Image 2"
[image9]: ./output_images/thresh_grad_3.png "Thresholded Image 3"
[image10]: ./output_images/thresh_grad_4.png "Thresholded Image 4"
[image11]: ./output_images/thresh_grad_5.png "Thresholded Image 5"
[image12]: ./output_images/thresh_grad_6.png "Thresholded Image 6"
[image13]: ./output_images/undist_source_points.png "Lane Source Points"
[image14]: ./output_images/warp_dest.png "Warped Lane Destination Points"
[image15]: ./output_images/reversed_with_box.png "Restore Lane Source Points with Inverse Matrix"
[image16]: ./output_images/warp_fit_poly.png "Fit Polynomial of Lane Pixels"
[image17]: ./output_images/test_pipeline_with_annotation.png "Processed Image with Annotation"
[video1]: ./project_video_out_full.mp4 "Video"

## 2. [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

All my implementations are in the IPython notebook named "P2.ipynb" in the root directory of this repository. I consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### 1. Camera Calibration & Image Distortion Correction

The code for this step is contained in the 2nd code cell of the IPython notebook "P2.ipynb".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `calibrate_camera()` function in the 2nd code block of the IPython notebook.  

### Pipeline (single images)

#### 1. Distortion Correction

Once the distortion coefficients are obtained from the calibration step, I use these distortion parameters as the inputs of `cv2.undistort()` function in the 4th notebook code block and test chessboard image on the left below using the and obtained this result image on the right below:

![Camera Calibration Result][image1]

In the 6th code block of the notebook, I apply the distortion correction to some of the test images as below: the left one is always the original image, the right one is the undistorted output image:
![Distortion Correction Test Image 1 ][image2]

![Distortion Correction Test Image 2 ][image3]

![Distortion Correction Test Image 3 ][image4]

It shows that after the image distortion correction, there are not much difference between the output images and the input images.

#### 2. Gradient and Color Thresholds

From 7th to the 14th code blocks of the IPython notebook are the codes to do color transforms, gradients on the given image and to create a thresholded binary image. Most of the functions in this section are inspired by the lecture contents and solutions to the quizzes. I refactored the original codes a little bit.

In my IPython notebook, I first tried to apply the sobel-x, sobel-y, magnitude, and the direction of the gradient thresholds and showed the results there from the 7th to the 10th code blocks.

The image below is the result of the 11th code block which combines sobel-x, sobel-y, magnitude, and direction of the gradient thresholds.

![Binary Image of Combining Gradient Thresholds][image5]

Applying the gradient thresholds makes the processed images in gray scale still look noisy. I hereby defined a function called `color_gradx_thresh()` combining the color and multiple gradient thresholds to generate a binary image below (related codes are in the 11th-12th code blocks of the IPython notebook).

![Binary Image of Combined Gradient Thresholds and Color Threshold][image6]

In the output image on the above right figure, the lane lines are more observable than the results of only applying the gradient threshold. However, the upper half of the image is all white, same color as the lane lines.

The root cause is because the color threshold parameter used above could not deal well with the lightness but only with the saturation. In the output image of `color_gradx_thresh()` function (see in the IPython notebook), the lane line color has the similar blue color as the sky. So we can tune the color threshold parameters to filter out the blue sky background and mark the lane line with a different color.

I referred other's article posted at https://zhuanlan.zhihu.com/p/35134563 (in Chinese), in which the author used a nice threshold method combining HLS, LAB, LUV color thresholds. Therefore, in my project, I
1. refactor the source codes into a neat function named `cvt_color_select()`, in the 13th code block, after understanding the original idea;
2. tune in the color parameters for HLS, LUV, and LAB thresholds to achiever better results compared to filter out more irrelative objects in the image compared with the original author's results (in the 14th code block).

![Thresholded Image 1][image7]

![Thresholded Image 2][image8]

![Thresholded Image 3][image9]

![Thresholded Image 4][image10]

![Thresholded Image 5][image11]

![Thresholded Image 6][image12]

#### 3. Perspective Transform

In the 15th code block, I obtained the perspective transform matrices `M` and `Minv` by calling OpenCV function called `cv2.getPerspectiveTransform()`.  This built-in function takes as inputs the source (`src`) and the destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 70, img_size[1]],
    [(img_size[0] / 2 + 95), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 203, 720      | 320, 720      |
| 1137, 720     | 960, 720      |
| 735, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Draw a red box around the lane lines][image13]

![The "bird-eye" view of the lane lines after perspective transform][image14]

![Apply the inverse transform matrix to restore the lane line from bird-eye view][image15]

In the IPython notebook, the 16th code block displays the undistorted lane, warped lane, and the gray-scale warped lane in parallel.

#### 4. Identify Lane-line Pixels and Fit With a Polynomial

To identify lane line pixels, the idea is first finding the points belonging to the left and right lanes using histogram peaks, then do sliding window polynomial fitting to those points.

Fortunately, I can simply re-use the codes from the lecture of "Implement Sliding Windows and Fit a Polynomial". Two functions: `finding_lane_pixels()` and `fit_polynomial()` are defined the 17th code block. Take the warped image from last step as the input, the picture below shows that the functions identifies the left and right lines successfully and plots a pile of green rectangles along each lane line, meanwhile, the computed 2nd order polynomial is in the yellow color.

![Fitting of the lane-line points][image16]

In the 19th code block, I also re-use the codes from the lecture quiz to define the `search_around_poly()` function. The difference is that here the inputs are the image and the arrays of fitting points of the left and right lines computed by the `fit_polynomial()` function.

#### 5. Calculated the Radius of Curvature and the Vehicle Offset.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image17]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
