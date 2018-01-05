# **Finding Lane Lines on the Road** 


The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

[image_gray]: ./report_images/gray.png "Grayscale"
[image_edges]: ./report_images/edges.png "Edges"
[image_hough]: ./report_images/combo_hough.png "Hough"
[image_hough2]: ./report_images/combo_hough2.png "Hough with image"
[image_left_side]: ./report_images/left_side.png "Left side points"
[image_right_side]: ./report_images/right_side.png "Right side points"
[image_masked_edges]: ./report_images/masked_edges.png "Masked edges"
[image_result]: ./report_images/result.png "Result"


---

### Reflection

### 1. Pipeline description.

My pipeline consisted of several steps. 
First, I converted the images to grayscale: 
![alt_text][image_gray]

Then I performed gaussian blur to get rid of pixel size noise.
After this I used Canny edge detection algorithm to get edges:
![alt_text][image_edges]

I cutted the edges to some region of interes which supposed to be part of picture with the road we mostly interested in:
![alt_text][image_masked_edges]

After this I performed Hough transform to detect lines consisted of points I've found via canny edge detection at previous step:
![alt_text][image_hough]
Same points but on source picture:
![alt_text][image_hough2]

In order to separate lines by their side i've sorted them just by middle vertical line. If points of line lies on the left side of the picture than I save it as a part of left line. Same for right line. So after this simple separation algorithm I got a tuple with 2 arrays: for left line and right line:
![alt_text][image_left_side]
![alt_text][image_right_side]

Next step was to find appropriate line approximation for these points, for each side. First, I put points belong to all lines I've got at Hough transform in order to increase accuracy, becouse Hough transform gives us only 2 point for line, but not the line points themself.
Last calculation stage was to calculate L2 approximation for all these points via default algorithm of line fitting. 
I did a couple transfromations to `cv2.fitLine` output to prepare line for image output and to fit it in our region of interes. As result I got:
![alt_text][image_result]


### 2. Shortcomings with my current pipeline


1. One potential shortcoming would be what would happen when the lines too curvy to be fitted just by straight line. Actually, that what happens on challenge video. 
2. cv2.fitLine() not so stable to outliers even with L1 or other metrics. Good alternative is RANSAC family algorithms, which works pretty nice with large amount of outliers and nonlinear models.
3. Calculation line points by using cv2.line() and grab them after from picture is not the best method. There is LineIterator in OpenCV realisation for C++, but not for Python. 
4. Canny edge detection parameters are nailed down inside program and not robust to changing conditions (light, weather, colors).
5. At this moment system is without any memory of line from previous frame what makes line unstable through video. 


### 3. Possible improvements to pipeline

1. To be more precise in line fitting we should use second or third order polynomials or splines.
2. Good alternative to cv2.fitLine() approximation is RANSAC family algorithms, which works pretty nice with large amount of outliers and nonlinear models.
3. There is a way to find pixel line points without intermediate plotting, as example here: https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator. It supposed to be faster way to get line points.
4. The wise way to choose canny edge detection parameters would be make some dependency off picture contrast. Supposed to be more robust.
5. Most obvious way to improve output lines is make them tied via frames. Simple exponential smoothing algorithm applied to right parameters of the line will help alot in term of stabilization.
