import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os


def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img, lines


# Python 3 has support for cool math symbols.
def weighted_img(img, initial_img, alpha=0.8, beta=1., l=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, l)


def line_angle(lines):
    """
    REWRITE
    lines are input data in format [x1, y1, x2, y2];
    result is angle between line and OX axis
    """
    angle = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle.append(np.arctan((y2 - y1) / (x2 - x1)))
    return np.array(angle)


def line_params(lines, y_intersect):
    """
    REWRITE
    lines are input data in format [x1, y1, x2, y2];
    result is angle between line and OX axis
    """
    params = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            b = (x1 * y2 - x2 * y1) / (x2 - x1)
            params.append([np.arctan(k), (y_intersect - b) / k])
    return np.array(params)


def get_x_coords(data, center):
    """
    function to find median around 2 peask in histogram
    :param data:
    :return:
    """
    (binvals, binrange) = np.histogram(data[:, 0])[:2]
    ind_max = np.argmax(binvals)
    binvals[ind_max] = 0
    ind_list = [ind_max, np.argmax(binvals)]
    return 0



# reading in an image
# сделать выше цикл чтения по всем картинкам

dir  = '../CarND-LaneLines-P1/test_images/'
list = os.listdir(dir)

image_path = os.path.join(dir, list[4])
print(list[2])
#image_path = '../CarND-LaneLines-P1/test_images/solidWhiteRight.jpg'

image = mpimg.imread(image_path)
shape = image.shape
# plt.imshow(image)
# plt.show()

# конвертируем в ЧБ
gray = grayscale(image)

# проходимся ядром гауса по картинке
kernel_size = 3
blur_gray = gaussian_blur(gray, kernel_size)


# берем границы алгоритмом Canny (было 50, 150)
low_threshold = 150
high_threshold = 300
edges = canny(blur_gray, low_threshold, high_threshold)

# plt.imshow(edges, cmap='Greys_r')
# plt.show()


# Отрезаем нужную область
imshape = image.shape
vertices = np.array([[(0,imshape[0]),(450, 320), (490, 320), (imshape[1],imshape[0])]], dtype=np.int32)
masked_edges = region_of_interest(edges, vertices)

# plt.imshow(masked_edges, cmap='Greys_r')
# plt.show()


# Проделываем преобразование Хафа и рисуем линии
rho = 2
theta = 1
threshold = 20
min_line_len = 10
max_line_gap = 15
line_img, lines = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

# plt.imshow(line_img)
# plt.show()

# Накладываем картинки
color_edges = np.dstack((edges, edges, edges))
combo = weighted_img(color_edges, line_img)

# plt.imshow(combo)
# plt.show()

# Тестовая часть
# Тут я тестирую и играюсь!


params = line_params(lines, -shape[0])

(angle_hist, angle_bins) = np.histogram(params[:, 0])[:2]
print(np.argmax(angle_hist))
angle_hist[0] = 0
print(np.argmax(angle_hist))


print(angle_hist)
print(angle_bins)


(x_hist, x_bins) = np.histogram(params[:, 1])[:2]
print(x_hist)
print(x_bins)

print(params[:, 1], shape[1])
print(params[:, 1] > shape[1]/2)

#np.argpartition(a, -4)[-4:]
#>>> a
#array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
#>>> ind = np.argpartition(a, -4)[-4:]
#>>> ind
#array([1, 5, 8, 0])
#>>> a[ind]
#array([4, 9, 6, 9])

#((25 < a) & (a < 100)).sum()