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

    # Convert to HSV for skin color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_positions = set()

    # Process hand contours
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # Filter out small contours (noise)
            hull = cv2.convexHull(contour)
            x, y, w, h = cv2.boundingRect(hull)
            aspect_ratio = h / w  

            if 0.7 < aspect_ratio < 1.8:  # Palm detection (not arm)
                palm_center = (x + w // 2, y + h // 2)
                col, row = palm_center[0] // step_x, palm_center[1] // step_y
                detected_positions.add((row, col))

    # Process face detection
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

    if pose_index >= len(pose_library):  # Once all poses are completed, freeze the frame
        cv2.putText(frame, f"Game Over! Final Score: {score}", (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Pose Matching Game', frame)
        cv2.waitKey(0)  # Wait indefinitely until a key is pressed
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
