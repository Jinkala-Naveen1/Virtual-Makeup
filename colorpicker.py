import cv2
import numpy as np
import mediapipe as mp

# Define facial feature landmarks for lips
upper_lip = [61, 185, 40, 39, 37, 0, 267, 269, 270, 408, 415, 272, 271, 268, 12, 38, 41, 42, 191, 78, 76]
lower_lip = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

def create_rgba_picker():
    """
    Creates an RGBA color picker window with trackbars for RGBA components.
    """
    window_name = "RGBA Picker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # Create trackbars for RGBA components
    cv2.createTrackbar("R", window_name, 0, 255, lambda x: None)
    cv2.createTrackbar("G", window_name, 0, 255, lambda x: None)
    cv2.createTrackbar("B", window_name, 0, 255, lambda x: None)
    cv2.createTrackbar("Alpha", window_name, 0, 255, lambda x: None)  # Alpha channel

    return window_name

def get_rgba_from_picker(window_name):
    """
    Retrieves the RGBA values from the trackbars in the RGBA color picker window.
    """
    # Ensure the window is still open
    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
        return None, None

    r = cv2.getTrackbarPos("R", window_name)
    g = cv2.getTrackbarPos("G", window_name)
    b = cv2.getTrackbarPos("B", window_name)
    a = cv2.getTrackbarPos("Alpha", window_name) / 255.0  # Normalize alpha to 0-1 range
    return (b, g, r), a

def detect_landmarks(src: np.ndarray, is_stream: bool = False):
    """
    Detects landmarks using MediaPipe FaceMesh.
    """
    mp_face_mesh = mp.solutions.face_mesh
    height, width, _ = src.shape
    with mp_face_mesh.FaceMesh(static_image_mode=not is_stream, max_num_faces=1, refine_landmarks=True) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(src, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0]  # Return the first face's landmarks

def normalize_landmarks(landmarks, height, width, feature_points):
    """
    Normalize the facial landmarks to the image size.
    """
    normalized_points = []
    for i in feature_points:
        keypoint = landmarks.landmark[i]
        normalized_x = int(keypoint.x * width)
        normalized_y = int(keypoint.y * height)
        normalized_points.append([normalized_x, normalized_y])
    return np.array(normalized_points)

def lip_mask(src: np.ndarray, points: np.ndarray, color: list, alpha: float):
    """
    Creates an RGBA mask for the lips based on the landmarks and applies the color with transparency.
    """
    mask = np.zeros_like(src, dtype=np.uint8)

    # Fill the mask with the RGBA color
    b, g, r = color
    rgba_color = [b, g, r, int(alpha * 255)]  # Convert alpha to 0-255 scale
    overlay = np.zeros((src.shape[0], src.shape[1], 4), dtype=np.uint8)
    cv2.fillPoly(overlay, [points], rgba_color)

    # Convert the source image to BGRA
    src_bgra = cv2.cvtColor(src, cv2.COLOR_BGR2BGRA)

    # Blend the mask with the source image using the alpha channel
    output = cv2.addWeighted(src_bgra, 1 - alpha, overlay, alpha, 0)

    return output

def apply_rgba_lip_color(image, picked_color, alpha):
    """
    Applies the picked RGBA color to the lips using facial landmarks.
    """
    # Detect landmarks
    landmarks = detect_landmarks(image)

    if landmarks is None:
        print("No face landmarks detected.")
        return image

    height, width, _ = image.shape

    # Normalize landmarks for the upper and lower lips
    lip_landmarks = normalize_landmarks(landmarks, height, width, upper_lip + lower_lip)

    # Create an RGBA mask for the lips and apply the picked color with transparency
    output = lip_mask(image, lip_landmarks, picked_color, alpha)

    return output

def main():
    # Load the image
    image = cv2.imread('input.jpg')  # Replace with your image path
    if image is None:
        print("Image not found. Please provide a valid image path.")
        return

    # Initialize the RGBA picker window
    window_name = create_rgba_picker()
    picked_color = (0, 0, 0)  # Default color
    alpha = 0.0  # Default alpha

    while True:
        # Get the selected RGBA color from the picker
        color, alpha = get_rgba_from_picker(window_name)
        if color is None:
            break  # If the window is closed, break the loop

        picked_color = color  # Update the picked color

        # Create a display with the selected color and alpha
        color_display = np.zeros((200, 300, 3), dtype=np.uint8)
        color_display[:] = picked_color
        cv2.rectangle(color_display, (50, 150), (250, 190), (200, 200, 200), -1)
        cv2.putText(color_display, "Select Color", (60, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Display the RGBA color selection window
        cv2.imshow("RGBA Picker", color_display)

        # Check for mouse click events on the "Select Color" button
        def mouse_click(event, x, y, flags, param):
            nonlocal picked_color
            if event == cv2.EVENT_LBUTTONDOWN:
                if 50 <= x <= 250 and 150 <= y <= 190:
                    cv2.destroyWindow(window_name)  # Close the RGBA picker window

        cv2.setMouseCallback("RGBA Picker", mouse_click)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Apply the picked RGBA color to the lips
    output = apply_rgba_lip_color(image.copy(), picked_color, alpha)

    # Display the final output
    cv2.imshow("Lips Makeup with RGBA", output)
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()

# Run the main function
main()
