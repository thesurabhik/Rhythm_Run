'''
import cv2
import numpy as np
import time

class Grid:
    def __init__(self):
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
    
    def update(self, detected_positions):
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
        for row, col in detected_positions:
            self.cells[row][col] = 1  # Mark the cell as occupied

    def matches(self, target, tolerance=0.75):  # Set tolerance to 75%
        matched_cells = 0
        total_cells = 0
        for i in range(3):
            for j in range(3):
                if self.cells[i][j] == target.cells[i][j]:
                    matched_cells += 1
                if self.cells[i][j] == 1 or target.cells[i][j] == 1:
                    total_cells += 1
        # Allow tolerance for matching (e.g., 75% overlap between target and current)
        return matched_cells / total_cells >= tolerance

pose_library = [
    Grid(), 
    Grid(),  
    Grid()   
]

pose_library[0].cells = [
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]

pose_library[1].cells = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

pose_library[2].cells = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
]

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

step_x, step_y = 640 // 3, 480 // 3  # Grid cell size
score = 0
pose_index = 0
pose_timer = time.time()

while pose_index < len(pose_library):  # Stop when last pose is reached
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_positions = set()

    for contour in contours:
        if cv2.contourArea(contour) > 3000:  # Filter out small contours (noise)
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)
            aspect_ratio = h / w  

            if 0.7 < aspect_ratio < 1.8:  # Palm detection (not arm)
                palm_center = (x + w // 2, y + h // 2)
                col, row = palm_center[0] // step_x, palm_center[1] // step_y
                detected_positions.add((row, col))

    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        col, row = face_center[0] // step_x, face_center[1] // step_y
        detected_positions.add((row, col))

    detected_grid = Grid()
    detected_grid.update(detected_positions)

    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y
            color = (200, 200, 200)

            if detected_grid.cells[i][j] == 1:  
                color = (0, 255, 255)

            if pose_library[pose_index].cells[i][j] == 1:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

    cv2.putText(frame, f"Pose {pose_index + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Score: {score}", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if time.time() - pose_timer >= 3:  # Change to 3 seconds
        if detected_grid.matches(pose_library[pose_index]):
            score += 10
            cv2.putText(frame, "Pose Matched! +10", (width // 2 - 100, height - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        pose_index += 1  # Move to next pose
        pose_timer = time.time()

    cv2.imshow('Pose Matching Game', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After completing all poses, display the final score and freeze the screen
if pose_index >= len(pose_library):
    cv2.putText(frame, f"Game Over! Final Score: {score}", (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Pose Matching Game', frame)
    cv2.waitKey(0)  # Wait indefinitely until a key is pressed

cap.release()
cv2.destroyAllWindows()


import cv2
import numpy as np

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to grab frame")
        break

    height, width, _ = frame.shape
    step_x, step_y = width // 3, height // 3  # Grid cell size

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of hands
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_positions = set()

    # Process detected hands
    for contour in contours:
        if cv2.contourArea(contour) > 3000:  # Filter small noise
            x, y, w, h = cv2.boundingRect(cv2.convexHull(contour))
            aspect_ratio = h / w
            
            if 0.7 < aspect_ratio < 1.8:  # Palm detection (not arm)
                palm_center = (x + w // 2, y + h // 2)
                col, row = palm_center[0] // step_x, palm_center[1] // step_y
                detected_positions.add((row, col))

    # Process detected faces
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        col, row = face_center[0] // step_x, face_center[1] // step_y
        detected_positions.add((row, col))

    # Draw 3x3 grid overlay
    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y
            color = (200, 200, 200)  # Default grid color

            if (i, j) in detected_positions:
                color = (0, 255, 255)  # Highlight detected cells
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Display the frame
    cv2.imshow('Hand & Head Detection', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
'''
import cv2
import numpy as np

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    if not ret:
        print("Failed to grab frame")
        break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Define 3x3 grid
    step_x, step_y = width // 3, height // 3
    highlighted_cells = set()

    # Convert to HSV for skin detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create skin mask
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find contours of hands
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for highlighting only detected hands within grid cells
    mask_highlight = np.zeros_like(frame)

    for contour in contours:
        if cv2.contourArea(contour) > 3000:  # Filter small noise
            hull = cv2.convexHull(contour)

            # Get bounding box for the hand
            x, y, w, h = cv2.boundingRect(hull)
            aspect_ratio = h / w  # Helps distinguish palm from arm
            
            # Ensure the detected region is not too tall (avoids wrist/arm detection)
            if 0.7 < aspect_ratio < 1.8:  # Palm is roughly square, arm is elongated
                hull_indices = cv2.convexHull(contour, returnPoints=False)
                defects = cv2.convexityDefects(contour, hull_indices)

                if defects is not None:
                    # Compute palm center
                    palm_center = (x + w // 2, y + h // 2)

                    # Determine which grid cell the palm is in
                    col, row = palm_center[0] // step_x, palm_center[1] // step_y
                    highlighted_cells.add((row, col))

                    # Compute the bounding box of the grid cell
                    grid_x1, grid_y1 = col * step_x, row * step_y
                    grid_x2, grid_y2 = (col + 1) * step_x, (row + 1) * step_y

                    # Create a mask for only the part of the hand inside the grid cell
                    mask_hand = np.zeros_like(frame)
                    cv2.drawContours(mask_hand, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)

                    # Apply the mask to the original frame but limit it within the detected grid cell
                    mask_hand = cv2.bitwise_and(mask_hand, mask_hand, mask=mask[:, :, np.newaxis])
                    mask_hand = mask_hand[grid_y1:grid_y2, grid_x1:grid_x2]  # Crop to grid cell

                    # Overlay the hand inside the detected grid cell
                    mask_highlight[grid_y1:grid_y2, grid_x1:grid_x2] = mask_hand

    # Process detected faces
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        col, row = face_center[0] // step_x, face_center[1] // step_y
        highlighted_cells.add((row, col))

        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    # Draw 3x3 grid overlay
    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y

            # Highlight grid cells with a face or hand
            if (i, j) in highlighted_cells:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)

    # Overlay the cropped hand image on the original frame
    frame = cv2.addWeighted(frame, 1, mask_highlight, 1, 0)

    # Display the frame
    cv2.imshow('Palm & Face Tracking with Grid', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
import cv2
import numpy as np
import time

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define a 3x3 Grid class for pose matching
class Grid:
    def __init__(self):
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
    
    def update(self, detected_positions):
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
        for row, col in detected_positions:
            self.cells[row][col] = 1  # Mark the cell as occupied

    def matches(self, target, tolerance=0.75):
        matched_cells = 0
        total_cells = 0
        for i in range(3):
            for j in range(3):
                if self.cells[i][j] == target.cells[i][j]:
                    matched_cells += 1
                if self.cells[i][j] == 1 or target.cells[i][j] == 1:
                    total_cells += 1
        return matched_cells / total_cells >= tolerance if total_cells > 0 else False

# Define target poses
pose_library = [
    Grid(), Grid(), Grid()
]
pose_library[0].cells = [
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]
pose_library[1].cells = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]
pose_library[2].cells = [
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
]

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

# Grid settings
step_x, step_y = 640 // 3, 480 // 3  # Adjust based on actual frame size
score = 0
pose_index = 0
pose_timer = time.time()

def detect_hands_and_face(frame):
    height, width, _ = frame.shape
    step_x, step_y = width // 3, height // 3
    detected_positions = set()
    
    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        col, row = face_center[0] // step_x, face_center[1] // step_y
        detected_positions.add((row, col))
    
    # Hand detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 3000:
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)
            aspect_ratio = h / w
            if 0.7 < aspect_ratio < 1.8:
                palm_center = (x + w // 2, y + h // 2)
                col, row = palm_center[0] // step_x, palm_center[1] // step_y
                detected_positions.add((row, col))
    
    return detected_positions

while pose_index < len(pose_library):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        print("Failed to grab frame")
        break
    
    detected_positions = detect_hands_and_face(frame)
    detected_grid = Grid()
    detected_grid.update(detected_positions)
    
    # Draw grid
    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y
            color = (200, 200, 200)
            if detected_grid.cells[i][j] == 1:
                color = (0, 255, 255)
            if pose_library[pose_index].cells[i][j] == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    
    # Scoring logic
    if time.time() - pose_timer >= 3:
        if detected_grid.matches(pose_library[pose_index]):
            score += 10  # Correct pose
            cv2.putText(frame, "Pose Matched! +10", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            score += 5  # Incorrect pose
            cv2.putText(frame, "Wrong Pose! +5", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        pose_index += 1
        pose_timer = time.time()
    
    cv2.putText(frame, f"Pose {pose_index + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Score: {score}", (480, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Pose Matching Game', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
