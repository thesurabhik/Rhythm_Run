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
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0]
]

pose_library[1].cells = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
]

pose_library[2].cells = [
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1]
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
                
                # Ensure the hand position is clipped to the bounds of the grid cell
                grid_x1, grid_y1 = col * step_x, row * step_y
                grid_x2, grid_y2 = (col + 1) * step_x, (row + 1) * step_y

                # Create a mask for the part of the hand inside the grid
                hand_in_grid_mask = np.zeros_like(frame)
                cv2.drawContours(hand_in_grid_mask, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
                hand_in_grid_mask = hand_in_grid_mask[grid_y1:grid_y2, grid_x1:grid_x2]

                # Check if any part of the hand touches the grid area and mark the cell
                if np.any(hand_in_grid_mask):  # If there's any part of the hand in the grid
                    detected_positions.add((row, col))

    # Process face detection
    for (x, y, w, h) in faces:
        face_center = (x + w // 2, y + h // 2)
        col, row = face_center[0] // step_x, face_center[1] // step_y
        detected_positions.add((row, col))

    detected_grid = Grid()
    detected_grid.update(detected_positions)

    # Prevent accessing index out of bounds
    if pose_index < len(pose_library):  # Ensure we are within the bounds of pose_library
        target_pose = pose_library[pose_index]
    else:
        target_pose = Grid()  # Default empty grid

    # Draw the grid
    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y
            color = (200, 200, 200)

            if detected_grid.cells[i][j] == 1:  
                color = (0, 255, 255)

            if target_pose.cells[i][j] == 1:  
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            # Add grid numbers to the center of each cell
            cv2.putText(frame, str(i * 3 + j + 1), (x1 + step_x // 3, y1 + step_y // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display which grid cells have been detected at the bottom left
    detected_numbers = [str(i * 3 + j + 1) for i, j in detected_positions]
    detected_text = "Detected: " + ", ".join(detected_numbers)
    cv2.putText(frame, detected_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    score = 0
    cv2.putText(frame, f"Pose {pose_index + 1}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Score: {score}", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if time.time() - pose_timer >= 5:  # Change to 3 seconds
        if detected_grid.matches(target_pose):
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
'''
import cv2
import numpy as np
import time



class Grid:
    def __init__(self):
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
    
    def update(self, detected_positions):
        # Reset the grid
        self.cells = [[0 for _ in range(3)] for _ in range(3)]
        # Update the grid based on detected positions
        for row, col in detected_positions:
            self.cells[row][col] = 1  # Mark the cell as occupied

    def get_grid_numbers(self):
        # Return the grid numbers (1-9) based on positions
        grid_numbers = []
        for i in range(3):
            for j in range(3):
                if self.cells[i][j] == 1:
                    grid_numbers.append(i * 3 + j + 1)  # Map grid positions to numbers
        return grid_numbers

    def matches(self, target_grid, detected_grid, tolerance=0.66):
        # Compare the detected grid with the target pose (only at least 2 out of 3)
        matched_cells = 0
        detected_cells = detected_grid.get_grid_numbers()
        for number in detected_cells:
            if number in target_grid.get_grid_numbers():
                matched_cells += 1
        return matched_cells >= 2

# Define target poses for the player (pose_library)
pose_library = [
    Grid(), 
    Grid(),  
    Grid(),
    Grid(),
    Grid(),
    Grid(),
    Grid()   
]

# Example poses with 1s indicating target grid cells (positions)
pose_library[0].cells = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0]
]

pose_library[1].cells = [
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0]
]

pose_library[2].cells = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
]

pose_library[3].cells = [
    [1, 0, 1],
    [0, 1, 0],
    [0, 0, 0]
]

pose_library[4].cells = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0]
]

pose_library[5].cells = [
    [0, 0, 1],
    [0, 1, 0],
    [0, 0, 1]
]

pose_library[6].cells = [
    [1, 0, 1],
    [0, 0, 0],
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
frame_count = 0  # Counter to delay hand and head detection

# Initialize background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# Pre-Game Message
ret, frame = cap.read()
frame = cv2.flip(frame, 1)

# Fill screen with black
frame[:] = (0, 0, 0)

# Show "Match the poses" for one second
cv2.putText(frame, "Match the poses", (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
cv2.imshow("Pose Matching Game", frame)
cv2.waitKey(1000)  # Wait for 1 second

# Countdown from 3 to 1
for i in range(3, 0, -1):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    # Fill screen with black
    frame[:] = (0, 0, 0)
    
    # Show countdown
    cv2.putText(frame, str(i), (300, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 5)
    
    cv2.imshow("Pose Matching Game", frame)
    cv2.waitKey(1000)  # Wait for 1 second

# Ensure the first pose is displayed fully before detecting
#pose_index = 0  # Set to first pose
#ret, frame = cap.read()
#frame = cv2.flip(frame, 1)

# Draw the first pose’s red grid before starting detection
#draw_pose_grid(frame, pose_library[pose_index])  # Assuming draw_pose_grid() handles red box drawing
#cv2.imshow("Pose Matching Game", frame)
#cv2.waitKey(2000)  # Display first pose for 2 seconds before detection starts

# Now start the main game loop with pose detection
start_time = time.time()  # Reset pose timer so detection starts fresh


while pose_index < len(pose_library):  # Stop when last pose is reached
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height, width, _ = frame.shape
    if not ret:
        print("Failed to grab frame")
        break

    # Apply background subtraction to isolate the foreground
    fg_mask = bg_subtractor.apply(frame)

    # Perform some morphology operations to clean up the foreground mask
    kernel = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Red boxes (grid) displayed first, showing the target pose
    target_pose = pose_library[pose_index]
    for i in range(3):
        for j in range(3):
            x1, y1 = j * step_x, i * step_y
            x2, y2 = (j + 1) * step_x, (i + 1) * step_y
            # Draw red boxes for the target pose
            if target_pose.cells[i][j] == 1:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red boxes for target pose

    # Once a few frames have passed, start the hand and head detection
    if frame_count > 10:  # Start detecting hand and head after 10 frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Convert to HSV for skin color detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Apply the foreground mask to filter out background noise
        mask = cv2.bitwise_and(mask, mask, mask=fg_mask)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_positions = set()

        # Process hand contours
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter out small contours (noise)
                hull = cv2.convexHull(contour)
                x, y, w, h = cv2.boundingRect(hull)
                aspect_ratio = h / w  

                if 0.7 < aspect_ratio < 1.8:  # Palm detection (not arm)
                    palm_center = (x + w // 2, y + h // 2)
                    col, row = palm_center[0] // step_x, palm_center[1] // step_y
                    
                    # Ensure the hand position is clipped to the bounds of the grid cell
                    grid_x1, grid_y1 = col * step_x, row * step_y
                    grid_x2, grid_y2 = (col + 1) * step_x, (row + 1) * step_y

                    # Create a mask for the part of the hand inside the grid
                    hand_in_grid_mask = np.zeros_like(frame)
                    cv2.drawContours(hand_in_grid_mask, [hull], -1, (255, 255, 255), thickness=cv2.FILLED)
                    hand_in_grid_mask = hand_in_grid_mask[grid_y1:grid_y2, grid_x1:grid_x2]

                    # Check if any part of the hand touches the grid area and mark the cell
                    if np.any(hand_in_grid_mask):  # If there's any part of the hand in the grid
                        detected_positions.add((row, col))

        # Process face detection
        for (x, y, w, h) in faces:
            face_center = (x + w // 2, y + h // 2)
            col, row = face_center[0] // step_x, face_center[1] // step_y
            detected_positions.add((row, col))

        # Limit the detected positions to only 3 cells
        detected_positions = set(list(detected_positions)[:3])  # Keep only the first 3 detected positions

        detected_grid = Grid()
        detected_grid.update(detected_positions)

        # Draw the detected positions
        for i in range(3):
            for j in range(3):
                if detected_grid.cells[i][j] == 1:  
                    x1, y1 = j * step_x, i * step_y
                    x2, y2 = (j + 1) * step_x, (i + 1) * step_y
                    # Draw the box around the detected positions
                    #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    # Add grid numbers to the center of each cell
                    #cv2.putText(frame, str(i * 3 + j + 1), (x1 + step_x // 3, y1 + step_y // 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display which grid cells have been detected at the bottom left
        detected_numbers = [str(i * 3 + j + 1) for i, j in detected_positions]
        detected_text = "Detected: " + ", ".join(detected_numbers)
        cv2.putText(frame, detected_text, (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.putText(frame, f"Pose {pose_index}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Score: {score}", (width - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # After 5 seconds, check if the detected pose matches the target pose
        if time.time() - pose_timer >= 4:  # 4-second time limit
            if target_pose.matches(target_pose, detected_grid):  # Matching poses
                score += 10
                cv2.putText(frame, "Pose Matched! +10 Points", (150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            pose_timer = time.time()  # Reset the timer
            pose_index += 1  # Move to the next pose

    frame_count += 1

    # Freeze the screen and display final score after the third pose is reached
    if pose_index >= len(pose_library):
        cv2.putText(frame, f"Final Score: {score}", (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
        cv2.putText(frame, "Game Over", (width // 2 - 150, height // 2 - 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)
        cv2.imshow("Pose Detection", frame)
        cv2.waitKey(0)  # Wait indefinitely until key press
        break

    cv2.imshow("Pose Detection", frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

''''
'# Freeze the screen and display final score after the third pose is reached
if pose_index >= len(pose_library):
    # Calculate the size of the final score text to ensure it's centered
    final_score_text = f"Final Score: {score}"
    text_size = cv2.getTextSize(final_score_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 4)[0]
    text_width, text_height = text_size

    # Set the position for the final score to be centered horizontally and vertically
    text_position_score = (width // 2 - text_width // 2, height // 2 - text_height // 2)

    # Display "Final Score" centered with bold letters
    cv2.putText(frame, final_score_text, text_position_score, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4, cv2.LINE_AA)
    
    # Display "Game Over" centered just below the final score
    text_position_game_over = (width // 2 - 150, height // 2 + 50)
    cv2.putText(frame, "Game Over", text_position_game_over, cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

    cv2.imshow("Pose Detection", frame)
    cv2.waitKey(0)  # Wait indefinitely until key press
    break'
'''
cap.release()
cv2.destroyAllWindows()
