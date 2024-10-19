import cv2
import numpy as np
from scipy.interpolate import interp1d
from skimage import color
import matplotlib.pyplot as plt

# Define landmarks for lips, face, and cheeks
upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
cheeks = [425, 205]

# Source color for eyeshadow
R, G, B = (102., 0., 51.)
inten = 0.8

# Points for eyeshadow from the file
lower_left_end = 5
upper_left_end = 11
lower_right_end = 16
upper_right_end = 22

# Function for interpolation
def inter(lx=[], ly=[], k1='quadratic'):
    unew = np.arange(lx[0], lx[-1] + 1, 1)
    f2 = interp1d(lx, ly, kind=k1)
    return f2, unew

def ext(a, b, i, x, y):
    indices = np.arange(int(a), int(b))
    x.extend(indices.tolist())
    y.extend([i] * len(indices))

def extleft(a, b, i, xleft, yleft):
    indices = np.arange(int(a), int(b))
    xleft.extend(indices.tolist())
    yleft.extend([i] * len(indices))

def extright(a, b, i, xright, yright):
    indices = np.arange(int(a), int(b))
    xright.extend(indices.tolist())
    yright.extend([i] * len(indices))

def apply_eyeshadow_effect(im):
    """
    Applies eyeshadow effect to the image.
    """
    # Load eyeshadow points from file
    file = np.loadtxt('pointeyeshadow.txt')
    points = np.floor(file)

    # Points for left and right eye
    point_down_x = np.array(points[:lower_left_end][:, 0])
    point_down_y = np.array(points[:lower_left_end][:, 1])
    point_up_x = np.array(points[lower_left_end:upper_left_end][:, 0])
    point_up_y = np.array(points[lower_left_end:upper_left_end][:, 1])
    point_down_x_right = np.array(points[upper_left_end:lower_right_end][:, 0])
    point_down_y_right = np.array(points[upper_left_end:lower_right_end][:, 1])
    point_up_x_right = np.array(points[lower_right_end:upper_right_end][:, 0])
    point_up_y_right = np.array(points[lower_right_end:upper_right_end][:, 1])

    # Offset adjustments for eyeshadow
    offset_left = max(point_down_y) - min(point_up_y)

    # Adjust array length to match `point_up_y` length
    point_up_y += offset_left * np.array([0.625, 0.3, 0.15, 0.1, 0.2, 0.1])
    point_down_y[0] += offset_left * 0.625

    offset_right = max(point_down_y_right) - min(point_up_y_right)

    # Adjust array length to match `point_up_y_right` length
    point_up_y_right += offset_right * np.array([0.625, 0.2, 0.1, 0.15, 0.3, 0.2])
    point_down_y_right[-1] += offset_right * 0.625

    # Create interpolation functions for left and right eyes
    l_l = inter(point_down_x, point_down_y, 'cubic')
    u_l = inter(point_up_x, point_up_y, 'cubic')
    l_r = inter(point_down_x_right, point_down_y_right, 'cubic')
    u_r = inter(point_up_x_right, point_up_y_right, 'cubic')

    # Prepare arrays for eyeshadow application
    x, y, xleft, yleft, xright, yright = [], [], [], [], [], []
    height, width = im.shape[:2]

    # Extend eyeshadow regions
    for i in range(int(l_l[1][0]), int(l_l[1][-1] + 1)):
        ext(u_l[0](i), l_l[0](i) + 1, i, x, y)
        extleft(u_l[0](i), l_l[0](i) + 1, i, xleft, yleft)

    for i in range(int(l_r[1][0]), int(l_r[1][-1] + 1)):
        ext(u_r[0](i), l_r[0](i) + 1, i, x, y)
        extright(u_r[0](i), l_r[0](i) + 1, i, xright, yright)

    # Ensure x and y are numpy arrays of integers
    x = np.array(x, dtype=int)
    y = np.array(y, dtype=int)

    # **Clamp the indices to be within the image bounds**
    x = np.clip(x, 0, height - 1)
    y = np.clip(y, 0, width - 1)

    # Convert image regions from RGB to LAB for color manipulation
    val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)

    # Modify LAB values to apply the eyeshadow color
    L1, A1, B1 = color.rgb2lab(np.array([R / 255., G / 255., B / 255.]).reshape(1, 1, 3)).reshape(3,)
    val[:, 0] += (L1 - np.mean(val[:, 0])) * inten
    val[:, 1] += (A1 - np.mean(val[:, 1])) * inten
    val[:, 2] += (B1 - np.mean(val[:, 2])) * inten

    # Create a blank image for blending the eyeshadow
    image_blank = np.zeros_like(im)
    image_blank[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255

    # Modify original image with eyeshadow
    im[x, y] = image_blank[x, y]

    # Apply Gaussian Blur and Erosion for better blending
    filter = np.zeros((height, width))
    cv2.fillConvexPoly(filter, np.array(np.c_[yleft, xleft], dtype='int32'), 1)
    cv2.fillConvexPoly(filter, np.array(np.c_[yright, xright], dtype='int32'), 1)
    filter = cv2.GaussianBlur(filter, (31, 31), 0)
    kernel = np.ones((12, 12), np.uint8)
    filter = cv2.erode(filter, kernel, iterations=1)

    # Alpha blending of the eyeshadow with the original image
    alpha = np.zeros([height, width, 3], dtype='float64')
    alpha[:, :, 0] = filter
    alpha[:, :, 1] = filter
    alpha[:, :, 2] = filter

    # Final blended output
    output = (alpha * im + (1 - alpha) * im).astype('uint8')

    return output

# Main apply_makeup function
def apply_makeup(src: np.ndarray, is_stream: bool, feature: str, show_landmarks: bool = False):
    """
    Takes in a source image and applies makeup effects onto it.
    """
    if src is None:
        print("Error: Image not loaded. Check the image path.")
        return None

    height, width, _ = src.shape

    if feature == 'lips':
        # Apply lipstick (simplified)
        mask = lip_mask(src, upper_lip + lower_lip, [153, 0, 157])
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)

    elif feature == 'blush':
        # Apply blush
        mask = blush_mask(src, cheeks, [153, 0, 157], 50)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)

    elif feature == 'eyeshadow':
        # Apply eyeshadow
        output = apply_eyeshadow_effect(src)

    else:
        # Default to foundation (gamma correction)
        skin_mask = mask_skin(src)
        output = np.where(src * skin_mask >= 1, gamma_correction(src, 1.75), src)

    if show_landmarks:
        # Optional: Visualize landmarks
        plot_landmarks(src)  # Assuming plot_landmarks is defined elsewhere

    return output

# Supporting functions (stubs for lip_mask, blush_mask, mask_skin, gamma_correction, plot_landmarks)

def lip_mask(src, points, color):
    """
    Apply a lip color mask to the lips.
    """
    mask = np.zeros_like(src)
    mask = cv2.fillPoly(mask, [np.array(points)], color)
    mask = cv2.GaussianBlur(mask, (7, 7), 5)
    return mask

def blush_mask(src, points, color, radius):
    """
    Apply blush to the cheeks.
    """
    mask = np.zeros_like(src)
    for point in points:
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)
    return mask

def mask_skin(src):
    """
    Create a skin mask for foundation application.
    """
    lower = np.array([0, 133, 77], dtype='uint8')
    upper = np.array([255, 173, 127], dtype='uint8')
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)
    skin_mask = cv2.inRange(dst, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)[..., np.newaxis]
    return (skin_mask / 255).astype("uint8")

def gamma_correction(src, gamma, coefficient=1):
    """
    Perform gamma correction for foundation.
    """
    dst = src / 255.0
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')
    return dst

# Placeholder for plot_landmarks function if needed.
def plot_landmarks(image):
    """
    Plot landmarks on the image (for debugging).
    """
    plt.imshow(image)
    plt.show()
