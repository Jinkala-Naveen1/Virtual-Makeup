import numpy as np
import cv2
from skimage import color
from scipy import interpolate
from matplotlib.pyplot import imshow, show, figure

# Nail polish color parameters (LAB)
Rg, Gg, Bg = (207., 40., 57.)
texture_input = 'texture2.jpg'  # Path to the texture image

# Function to get boundary points using spline interpolation
def get_boundary_points(x, y):
    tck, u = interpolate.splprep([x, y], s=0, per=1)  # s=0 ensures smoothness
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

# Load input image and texture image using cv2
im = cv2.imread('nail_inp.jpg', cv2.IMREAD_UNCHANGED).astype(np.float64)  # Nail input image
texture = cv2.imread(texture_input, cv2.IMREAD_UNCHANGED).astype(np.float64)  # Texture image

# Function to tile the texture over the region
def tile_texture_to_fit(texture, region_shape):
    texture_height, texture_width = texture.shape[:2]
    region_height, region_width = region_shape

    tile_y = np.ceil(region_height / texture_height).astype(int)
    tile_x = np.ceil(region_width / texture_width).astype(int)

    tiled_texture = np.tile(texture, (tile_y, tile_x, 1))

    return tiled_texture[:region_height, :region_width, :]

# Apply LAB color for nail polish and texture overlay
def apply_nail_polish(x, y, r=Rg, g=Gg, b=Bg, texture=None):
    val = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
    L1, A1, B1 = color.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3,)
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    val[:, 0] = np.clip(val[:, 0] + ll, 0, 100)
    val[:, 1] = np.clip(val[:, 1] + aa, -127, 128)
    val[:, 2] = np.clip(val[:, 2] + bb, -127, 128)
    im[x, y] = color.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255

    if texture is not None:
        apply_texture(x, y, texture)

# Apply texture overlay in LAB space
def apply_texture(x, y, texture):
    xmin, ymin = np.amin(x), np.amin(y)
    X = (x - xmin).astype(int)
    Y = (y - ymin).astype(int)

    region_height, region_width = len(np.unique(Y)), len(np.unique(X))

    tiled_texture = tile_texture_to_fit(texture, (region_height, region_width))

    X = X % tiled_texture.shape[1]
    Y = Y % tiled_texture.shape[0]

    val1 = color.rgb2lab((tiled_texture[Y, X] / 255.).reshape(len(X), 1, 3)).reshape(len(X), 3)
    val2 = color.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)

    L, A, B = np.mean(val2[:, 0]), np.mean(val2[:, 1]), np.mean(val2[:, 2])
    val2[:, 0] = np.clip(val2[:, 0] - L + val1[:, 0], 0, 100)
    val2[:, 1] = np.clip(val2[:, 1] - A + val1[:, 1], -127, 128)
    val2[:, 2] = np.clip(val2[:, 2] - B + val1[:, 2], -127, 128)

    im[x, y] = color.lab2rgb(val2.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255

# Load boundary points (nail coordinates)
try:
    points = np.loadtxt('nailpoint')
except Exception as e:
    print(f"Error loading nailpoint file: {e}")
    exit(1)

# Apply nail polish and texture to each segment of the nail
for i in range(0, len(points), 12):
    x, y = points[i:i+12, 0], points[i:i+12, 1]
    x, y = get_boundary_points(x, y)
    x, y = get_interior_points(x, y)
    apply_nail_polish(x, y, r=207, g=40, b=57, texture=texture)

# Prepare the output image
im = np.clip(im, 0, 255).astype(np.uint8)  # Ensure values are valid for uint8
figure()
imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))  # Convert BGR (OpenCV) to RGB for matplotlib
cv2.imwrite('output_texture_nail_polish.jpg', im)  # Save the final image
show()
