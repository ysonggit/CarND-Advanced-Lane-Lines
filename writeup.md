
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
[image18]: ./output_images/warp_search_around_poly.png "Search Around Polynomial"
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

In the 19th code block, I also re-use the codes from the lecture quiz to define the `search_around_poly()` function which uses the previous polynomial to skip the sliding window. So that here the inputs are the image and the arrays of fitting points of the left and right lines computed from the prior step, namely, the `fit_polynomial()` function.

In the figure below, assume that the fitting points of lane lines are given, the function `search_around_poly()` fits the polynomial in terms of these points, instead of using sliding window method, to improve the algorithm efficiency.

![Fit polynomial with previous lines][image18]

#### 5. Calculated the Radius of Curvature and the Vehicle Offset.

The radius of the lane's curvature is computed in terms of the polynomial equations given in the lecture. So I re-use most of the codes I had done for the lecture quiz of "Measuring Curvature II".

Because the original solution uses fake data representing lane-line pixels, the change in the 20th code block is to update the function to let it take actual data from inputs.

Meanwhile, the original solution returns only curvatures of the polynomial functions from the left and right lines. I hereby add one more step to compute the average curvature of the left and right curvatures for the final display.

The codes in the 20th block computes the offset, that is, the lane center from the center of the image. I referred to the `calculate_curv_and_pos()` function in the article [2]. The original author used this one function to return both curvature and offset. I like the author's idea of computing offset because he not only finds both the offset values but also the relative positions, left or right, between the car center and the lane center.

My changes are:
- Re-factor the offset computation as a single function. I believe this improves the pipeline design because curvature and offset are two independent factors, there is not much correlation between these two values.
- Remove the hard-coded parameters in the original codes to make the function be more robust and adaptive to different image resolutions.

In the 25th code block, I manually check if the computed curvature and offset of the example image are reasonable values, according to the project's guideline: "the radius of that circle is approximately 1 km".

#### 6. Result Image

As the line boundaries are computed according to the warped image, I apply the computed inverse transform matrix `Minv` to the warp image and project the detected lane boundaries back to the original undistorted image.

The 22nd code block defines the function `draw_area()` to plot the identified lane lines and lane area. The 23rd code block defines the function `annotation()` to annotate the image with the road curvature and the vehicle offset.

![Drivable Area Plotted Back to the Image][image17]

The image above shows an example of the result image. The left lane is red, the right lane is blue, and the drivable region of the lane is highlighted with green color. Meanwhile, when plotting the image, I invoke the `annotation()` function to display the curvature and offset information on the it.

---

### Pipeline (video)

To implement the entire image processing pipeline, in stead of using the `Line()` class suggested in the project guidelines, I referred to Huijing Huang's work [3]. The advantages of his work are:

1. It is easy to integrate the above mentioned methods into the Line class;
2. The sanity check is well defined along the class.

My `Line()` class is defined in the 24th code block and my changes are:
- As I already defined the most import functions above such as `search_around_poly()`, `fit_polynomial()`, `measure_curvature()`, `measure_offset()`, etc., I simply use those functions to replace Huang's original methods which functions the same.
- I fixed a bug in Huang's original code:
```python
if (self.detected):
      last_left_fit = self.curr_left_fit[-1]
      last_right_fit = self.curr_right_fit[-1]
```
this part will throw an exception when the array of fitting points, `curr_left_fit` or `curr_right_fit` are empty. So I added a sanity check to handle this case.

The pipeline implementation is in the 26th code block of the notebook. I test the pipeline with the project video and the result looks great.

Here's a [link to my video result](./project_video_out_full.mp4)

---

### Discussion

#### Problems/issues I had

There are two major difficulties I encountered when I work on this project.

1. How to process the input color image to show clearly the lane lines in an output binary image.
At the beginning, I simply apply the HLS thresholding and gradient threshold methods obtained from the lecture only, the result shows that this idea could not identify the lane line correctly most of the time. Therefore, I refer to the idea introduced in the references [1] and [2], and consider the different color spaces: HLS, LAB, and LUV, at the same time, to improve the robustness of the algorithm when handling the lighting and shadows in the given video.

2. Another difficulty I spent most of my time is to construct the `Line` class with data Sanity Check and Look-Ahead Filter. I refer to Huijing Huang's solution on his Github [3]
, because his class structure is well defined, very readable, so I can easily integrate the key line detecting and fitting functions of mine. Meanwhile, I fixed a bug in his codes.

#### Where will your pipeline likely fail?

The pitfall of my implementation is that the pipeline will fail when the lane has more than one curves in one image, for example, an "S"-like road ahead of the car.

I test my pipeline with a short clip of the challenge video, here is the [link of the result video](./challenge_video_out_clip.mp4). In the result video, the drivable lane area is not plotted correctly when the "S"-like curves appears in the camera. I consider the main reason is that the polynomial fitting function `fit_polynomial()` fails in this scenario because the it only tries to fit a 2nd order polynomial equation, which is a parabola. However, the lane's shape in this scenario is more like a trigonometric function, such as cosine.

#### What could you do to make it more robust?

Although I refactored the codes from the lectures and the open sources when I work on my implementations, my implementation still have many hard-coded parameters and implicit assumptions. To make my algorithm more robust, using dynamic parameters as much as possible could be my next improvement. For example,

1. When computing the perspective transform, I used hard-coded source and destination points. These values are good to use here but may be adjusted when the camera model is different.

2. When doing the gradient thresholding, I also used many hard-coded parameters. Obviously, there parameters work well in this scenario which has a good lighting condition. However, it may not work if the lighting condition changes, say, in a cloudy/raining day, or the evening before the sunset.


## References

1. [CIELAB color space wikipedia](https://en.wikipedia.org/wiki/CIELAB_color_space)
2. [Lane Finding (车道线检测)](https://zhuanlan.zhihu.com/p/35134563)
3. [Huijing Huang' Github](https://github.com/dmsehuang/CarND-Advanced-Lane-Lines)
