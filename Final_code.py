import cv2
import numpy as np
from scipy.interpolate import interp1d
from skimage import io, color
import matplotlib.pyplot as plt
from landmarks import detect_landmarks, normalize_landmarks, plot_landmarks
from mediapipe.python.solutions.face_detection import FaceDetection
import mediapipe as mp
from scipy import interpolate
from skimage.transform import resize

# Define facial feature landmarks
upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
cheeks = [425, 205]

# Nail polish parameters
Rg, Gg, Bg = (207., 40., 57.)

# Eyeshadow parameters
R, G, B = (102., 0., 51.)  # Eyeshadow RGB color

# Apply makeup effects
def apply_makeup(src: np.ndarray, is_stream: bool, feature: str, show_landmarks: bool = False):
    """
    Takes in a source image and applies effects onto it.
    """
    # Ensure the source image is in uint8 format before using cv2 functions
    if src.dtype != np.uint8:
        src = np.clip(src, 0, 255).astype(np.uint8)  # Convert float64 to uint8
    
    ret_landmarks = None
    if feature != 'nail_polish':
        ret_landmarks = detect_landmarks(src, is_stream)
    
        # Check if landmarks were detected, but only if the feature isn't nail polish
        if ret_landmarks is None and feature != 'nail_polish':
            raise ValueError("No face landmarks were detected. Please check the input image or ensure a face is visible.")

    height, width, _ = src.shape
    feature_landmarks = None
    
    if feature == 'lips':
        # Use FaceMesh's landmarks for lips
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, upper_lip + lower_lip)
        mask = lip_mask(src, feature_landmarks, [255, 0, 0])
        output = cv2.addWeighted(src, 1.0, mask, 0.4, 0.0)

    elif feature == 'blush':
        feature_landmarks = normalize_landmarks(ret_landmarks, height, width, cheeks)
        mask = blush_mask(src, feature_landmarks, [255, 0, 0], 150)
        output = cv2.addWeighted(src, 1.0, mask, 0.3, 0.0)

    elif feature == 'eyeshadow':  # Eyeshadow application
         output = apply_eyeshadow(src, ret_landmarks, {"EYESHADOW_LEFT": [R, G, B], "EYESHADOW_RIGHT": [R, G, B]})
    

    elif feature == 'eyeliner':  # Eyeliner application
        output = apply_eyeliner(src, ret_landmarks, {"EYELINER_LEFT": [139, 0, 0], "EYELINER_RIGHT": [139, 0, 0]})

    elif feature == 'eyebrows':  # Eyebrow application
        output = apply_eyebrows(src, ret_landmarks, {"EYEBROW_LEFT": [19, 69, 139], "EYEBROW_RIGHT": [19, 69, 139]})

    elif feature == 'nail_polish':  # Nail polish application
        try:
            points = np.loadtxt('nailpoint')
        except FileNotFoundError:
            raise FileNotFoundError('The nailpoint file is missing or cannot be loaded.')

        # Load texture image

        texture_input = 'texture.jpg'
        texture = cv2.imread(texture_input, cv2.IMREAD_UNCHANGED)
        if texture is None:
            raise FileNotFoundError(f"The texture image '{texture_input}' could not be loaded.")
        
        src=src.astype(np.float64)
        texture = texture.astype(np.float64)  # Convert texture to float64 if necessary for blending operations

        # Loop through each nail's boundary and apply polish and texture
        for i in range(0, len(points), 12):
            x, y = points[i:i+12, 0], points[i:i+12, 1]
            x, y = get_boundary_points(x, y)
            x, y = get_interior_points(x, y)
            src = apply_nail_polish(src, x, y, Rg, Gg, Bg, texture)

        # if src.dtype != np.uint8:
        #     src = np.clip(src, 0, 255).astype(np.uint8)

        output = src  # Return the modified image

    else:  # Defaults to Foundation 
        skin_mask = mask_skin(src)
        output = np.where(src * skin_mask >= 1, gamma_correction(src, 1.75), src)

    if show_landmarks and feature_landmarks is not None:
        plot_landmarks(src, feature_landmarks, True)

    return output


# Function to detect landmarks using MediaPipe
mp_face_mesh = mp.solutions.face_mesh

def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    """
    Detects landmarks using MediaPipe FaceMesh.
    """
    if src.dtype != np.uint8:
        src = np.clip(src, 0, 255).astype(np.uint8)

    height, width, _ = src.shape
    with mp_face_mesh.FaceMesh(static_image_mode=not is_stream, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        face_landmarks = results.multi_face_landmarks[0]  # Get the first face's landmarks
        return face_landmarks  # Return the landmarks object


def normalize_landmarks(landmarks, height, width, feature_points):
    """
    Normalize the facial landmarks provided by MediaPipe FaceMesh to the image size.
    """
    normalized_points = []
    
    for i in feature_points:
        # Access the landmark and convert it to image coordinates
        keypoint = landmarks.landmark[i]
        normalized_x = int(keypoint.x * width)
        normalized_y = int(keypoint.y * height)
        normalized_points.append([normalized_x, normalized_y])

    return np.array(normalized_points)


# Lip mask
def lip_mask(src: np.ndarray, points: np.ndarray, color: list):
    mask = np.zeros_like(src)  # Create a mask
    mask = cv2.fillPoly(mask, [points], color)  # Mask for the required facial feature
    mask = cv2.GaussianBlur(mask, (7, 7), 5)  # Blurring the region for a natural effect
    return mask

# Blush mask
def blush_mask(src: np.ndarray, points: np.ndarray, color: list, radius: int):
    mask = np.zeros_like(src)  # Mask for cheeks
    for point in points:
        mask = cv2.circle(mask, point, radius, color, cv2.FILLED)
        x, y = point[0] - radius, point[1] - radius  # Get top-left of the mask
        mask[y:y + 2 * radius, x:x + 2 * radius] = vignette(mask[y:y + 2 * radius, x:x + 2 * radius], 10)  # Vignette effect
    return mask

# Vignette effect
def vignette(src: np.ndarray, sigma: int):
    height, width, _ = src.shape
    kernel_x = cv2.getGaussianKernel(width, sigma)
    kernel_y = cv2.getGaussianKernel(height, sigma)
    kernel = kernel_y * kernel_x.T
    mask = kernel / kernel.max()
    blurred = cv2.convertScaleAbs(src.copy() * np.expand_dims(mask, axis=-1))
    return blurred

#  Gamma correction for brightness and contrast control
def gamma_correction(src: np.ndarray, gamma: float, coefficient: int = 1):
    dst = src.copy()
    dst = dst / 255.  # Convert to float
    dst = coefficient * np.power(dst, gamma)
    dst = (dst * 255).astype('uint8')  # Convert back to uint8
    return dst

#  Mask skin for makeup foundation or skin adjustments
def mask_skin(src: np.ndarray):
    lower = np.array([0, 133, 77], dtype='uint8')  # Lower bound for skin color in YCrCb
    upper = np.array([255, 173, 127], dtype='uint8')  # Upper bound for skin color
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2YCR_CB)  # Convert to YCrCb color space
    skin_mask = cv2.inRange(dst, lower, upper)  # Create skin mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)[..., np.newaxis]  # Dilate to fill in small gaps
    if skin_mask.ndim != 3:
        skin_mask = np.expand_dims(skin_mask, axis=-1)
    return (skin_mask / 255).astype("uint8")

##Eyeshadow

# Eyeshadow application function

def apply_eyeshadow(image: np.array, face_landmarks: dict, colors_map: dict) -> np.array:
    """
    Applies eyeshadow to an image based on the detected facial landmarks.

    image: np.array - The input image to apply eyeshadow on.
    face_landmarks: NormalizedLandmarkList - The face landmarks from MediaPipe FaceMesh.
    colors_map: dict - A dictionary mapping facial features (eyeshadow) to BGR color values.

    Returns:
    np.array - The image with eyeshadow applied.
    """
    # Eyeshadow facial landmarks
    eyeshadow_left = [226, 247, 30, 29, 27, 28, 56, 190, 243, 173, 157, 158, 159, 160, 161, 246, 33, 130, 226]
    eyeshadow_right = [463, 414, 286, 258, 257, 259, 260, 467, 446, 359, 263, 466, 388, 387, 386, 385, 384, 398, 362, 463]

    # Convert the face_landmarks (NormalizedLandmarkList) into a dictionary of pixel coordinates
    landmark_dict = {}
    height, width, _ = image.shape
    for idx, landmark in enumerate(face_landmarks.landmark):
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        if landmark_px:
            landmark_dict[idx] = landmark_px

    # Create a mask with zeros (black image)
    mask = np.zeros_like(image)

    # Get corresponding colors for the eyeshadow
    eyeshadow_left_color = colors_map.get("EYESHADOW_LEFT", [0, 100, 0])  # Default dark green
    eyeshadow_right_color = colors_map.get("EYESHADOW_RIGHT", [0, 100, 0])  # Default dark green

    # Apply eyeshadow to left eye
    if all(idx in landmark_dict for idx in eyeshadow_left):
        left_eye_points = np.array([landmark_dict[idx] for idx in eyeshadow_left])
        cv2.fillPoly(mask, [left_eye_points], eyeshadow_left_color)

    # Apply eyeshadow to right eye
    if all(idx in landmark_dict for idx in eyeshadow_right):
        right_eye_points = np.array([landmark_dict[idx] for idx in eyeshadow_right])
        cv2.fillPoly(mask, [right_eye_points], eyeshadow_right_color)

    # Smooth the mask with Gaussian Blur for a soft look
    mask = cv2.GaussianBlur(mask, (7, 7), 4)

    # Blend the eyeshadow mask with the original image
    output = cv2.addWeighted(image, 1.0, mask, 0.3, 1.0)  # Adjust the weight to control intensity of eyeshadow

    return output

def apply_eyeliner(image: np.array, face_landmarks: dict, colors_map: dict) -> np.array:
    """
    Applies eyeliner to an image based on the detected facial landmarks.

    image: np.array - The input image to apply eyeliner on.
    face_landmarks: dict - A dictionary containing the facial landmark coordinates.
    colors_map: dict - A dictionary mapping facial features (eyeliner) to BGR color values.

    Returns:
    np.array - The image with eyeliner applied.
    """
    # Eyeliner facial landmarks
    eyeliner_left = [243, 112, 26, 22, 23, 24, 110, 25, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243]
    eyeliner_right = [463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 359, 446, 255, 339, 254, 253, 252, 256, 341, 463]

    # Convert the face_landmarks (NormalizedLandmarkList) into a dictionary of pixel coordinates
    landmark_dict = {}
    height, width, _ = image.shape
    for idx, landmark in enumerate(face_landmarks.landmark):
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        if landmark_px:
            landmark_dict[idx] = landmark_px

    # Create a mask with zeros (black image)
    mask = np.zeros_like(image)

    # Get corresponding colors for the eyeliner
    eyeliner_left_color = colors_map.get("EYELINER_LEFT", [139, 0, 0])  # Dark blue (default)
    eyeliner_right_color = colors_map.get("EYELINER_RIGHT", [139, 0, 0])  # Dark blue (default)

    # Apply eyeliner to left eye
    if all(idx in landmark_dict for idx in eyeliner_left):
        left_eye_points = np.array([landmark_dict[idx] for idx in eyeliner_left])
        cv2.polylines(mask, [left_eye_points], isClosed=False, color=eyeliner_left_color, thickness=2)

    # Apply eyeliner to right eye
    if all(idx in landmark_dict for idx in eyeliner_right):
        right_eye_points = np.array([landmark_dict[idx] for idx in eyeliner_right])
        cv2.polylines(mask, [right_eye_points], isClosed=False, color=eyeliner_right_color, thickness=2)

    # Blend the eyeliner mask with the original image
    output = cv2.addWeighted(image, 1.0, mask, 0.5, 1.0)  # Adjust the weight to control the intensity of the eyeliner

    return output


def apply_eyebrows(image: np.array, face_landmarks: dict, colors_map: dict) -> np.array:
    """
    Applies color to the eyebrows based on the detected facial landmarks.

    image: np.array - The input image to apply eyebrow color on.
    face_landmarks: dict - A dictionary containing the facial landmark coordinates.
    colors_map: dict - A dictionary mapping facial features (eyebrows) to BGR color values.

    Returns:
    np.array - The image with eyebrows colored.
    """
    # Eyebrow facial landmarks
    eyebrow_left = [55, 107, 66, 105, 63, 70, 46, 53, 52, 65, 55]
    eyebrow_right = [285, 336, 296, 334, 293, 300, 276, 283, 295, 285]

    # Convert the face_landmarks (NormalizedLandmarkList) into a dictionary of pixel coordinates
    landmark_dict = {}
    height, width, _ = image.shape
    for idx, landmark in enumerate(face_landmarks.landmark):
        landmark_px = mp.solutions.drawing_utils._normalized_to_pixel_coordinates(
            landmark.x, landmark.y, width, height
        )
        if landmark_px:
            landmark_dict[idx] = landmark_px

    # Create a mask with zeros (black image)
    mask = np.zeros_like(image)

    # Get corresponding colors for the eyebrows
    eyebrow_left_color = colors_map.get("EYEBROW_LEFT", [19, 69, 139])  # Default dark brown
    eyebrow_right_color = colors_map.get("EYEBROW_RIGHT", [19, 69, 139])  # Default dark brown

    # Apply eyebrow color to the left eyebrow
    if all(idx in landmark_dict for idx in eyebrow_left):
        left_eyebrow_points = np.array([landmark_dict[idx] for idx in eyebrow_left])
        cv2.fillPoly(mask, [left_eyebrow_points], eyebrow_left_color)

    # Apply eyebrow color to the right eyebrow
    if all(idx in landmark_dict for idx in eyebrow_right):
        right_eyebrow_points = np.array([landmark_dict[idx] for idx in eyebrow_right])
        cv2.fillPoly(mask, [right_eyebrow_points], eyebrow_right_color)

    # Blend the eyebrow mask with the original image
    output = cv2.addWeighted(image, 1.0, mask, 0.5, 1.0)  # Adjust the weight to control the intensity of the eyebrows

    return output

#Nail Polish

def tile_texture_to_fit(texture, region_shape):
    """
    Tiles the texture to fit the size of the region if the region is larger than the texture.
    """
    texture_height, texture_width = texture.shape[:2]
    region_height, region_width = region_shape

    # Calculate how many times we need to repeat the texture to cover the region
    tile_y = np.ceil(region_height / texture_height).astype(int)
    tile_x = np.ceil(region_width / texture_width).astype(int)

    # Repeat the texture to cover the region size
    tiled_texture = np.tile(texture, (tile_y, tile_x, 1))

    # Crop the tiled texture to exactly fit the region size
    return tiled_texture[:region_height, :region_width, :]
# Function to get boundary points using spline interpolation

def get_boundary_points(x, y):
    tck, u = interpolate.splprep([x, y], s=0, per=1)  # s=0 ensures the curve passes through the points
    unew = np.linspace(u.min(), u.max(), 1000)  # Interpolate with 1000 points for smoothness
    xnew, ynew = interpolate.splev(unew, tck, der=0)
    tup = np.c_[xnew.astype(int), ynew.astype(int)].tolist()
    coord = list(set(tuple(map(tuple, tup))))  # Remove duplicate points
    coord = np.array([list(elem) for elem in coord])
    return np.array(coord[:, 0], dtype=np.int32), np.array(coord[:, 1], dtype=np.int32)

# Function to get interior points of a region
def get_interior_points(x, y):
    nailx, naily = [], []

    def ext(a, b, i):
        a, b = round(a), round(b)
        nailx.extend(np.arange(a, b, 1).tolist())
        naily.extend((np.ones(b - a) * i).tolist())

    x, y = np.array(x), np.array(y)
    xmin, xmax = np.amin(x), np.amax(x)
    xrang = np.arange(xmin, xmax + 1, 1)

    for i in xrang:
        ylist = y[np.where(x == i)]
        ext(np.amin(ylist), np.amax(ylist), i)

    return np.array(nailx, dtype=np.int32), np.array(naily, dtype=np.int32)

# Apply LAB color for nail polish and texture overlay
# Apply LAB color for nail polish and texture overlay
# Apply LAB color for nail polish and texture overlay
def apply_nail_polish(im, x, y, r=Rg, g=Gg, b=Bg, texture=None):
    """
    Apply nail polish and optionally blend texture over the nail region.
    """
    val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3,)
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)     # Clamp L to [0, 100]
    val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)  # Clamp A to [-127, 128]
    val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)  # Clamp B to [-127, 128]
    
    # Convert LAB to RGB and ensure values are clamped between 0 and 255
    rgb_values = np.clip(color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255, 0, 255).astype(np.uint8)

    # Apply the adjusted RGB values to the image
    im[x, y] = rgb_values

    # Apply texture overlay if texture is provided
    if texture is not None:
        im = apply_texture(im, x, y, texture)

    return im


# Apply texture overlay in LAB space
def apply_texture(im, x, y, texture):
    xmin, ymin = np.amin(x), np.amin(y)
    X = (x - xmin).astype(int)
    Y = (y - ymin).astype(int)

    # Get the dimensions of the nail region
    region_height, region_width = len(np.unique(Y)), len(np.unique(X))

    # Tile the texture to fit the region
    tiled_texture = tile_texture_to_fit(texture, (region_height, region_width))

    # Map the nail region indices to the texture indices
    X = X % tiled_texture.shape[1]  # Ensure X is within the bounds of the texture
    Y = Y % tiled_texture.shape[0]  # Ensure Y is within the bounds of the texture

    # Convert both images to LAB color space
    val1 = color.rgb2lab((tiled_texture[Y, X] / 255.).reshape(len(X), 1, 3)).reshape(len(X), 3)
    val2 = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)

    L, A, B = np.mean(val2[:, 0]), np.mean(val2[:, 1]), np.mean(val2[:, 2])
    val2[:, 0] = np.clip(val2[:, 0] - L + val1[:, 0], 0, 100)
    val2[:, 1] = np.clip(val2[:, 1] - A + val1[:, 1], -127, 128)
    val2[:, 2] = np.clip(val2[:, 2] - B + val1[:, 2], -127, 128)

    im[x, y] = np.clip(color.lab2rgb(val2.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255, 0, 255).astype(np.uint8)

    return im
