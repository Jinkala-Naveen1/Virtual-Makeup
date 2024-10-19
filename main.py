import cv2
import numpy as np
from Final_code import apply_makeup

# Load the input image in color mode
image = cv2.imread('input.jpg', cv2.IMREAD_COLOR)

# Check if the image was loaded successfully
if image is None:
    raise ValueError("Input image could not be loaded. Check the file path.")

# Apply makeup effect (e.g., 'lips')
try:
    output = apply_makeup(image, False, 'lips', False)
except Exception as e:
    raise RuntimeError(f"An error occurred during makeup application: {e}")

# Ensure the output is in uint8 format
output = output.astype(np.uint8)

# Display the original and processed images
cv2.imshow("Original", image)
cv2.imshow("Feature", output)

# Wait for a key press to close the windows
cv2.waitKey(0)
cv2.destroyAllWindows()
