import cv2

# Load the pre-trained Haar Cascade face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start the video capture (0 for the built-in Mac camera)
cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    height, width, _ = frame.shape
    if not ret:
        print("Failed to grab frame")
        break

    # Convert the image to grayscale (Haar Cascade works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    red = (0, 0, 255)    # Upper region
    yellow = (0, 255, 255)  # Middle region
    green = (0, 255, 0)  # Lower region
    
    upper_limit = height // 3
    middle_limit = 2 * height // 3

    # Draw rectangles around the faces
    for (x, y, w, h) in faces:
        if w < 50 or h < 50:  # You can adjust these values
            continue
        face_center_y = y + h // 2

        

        # Determine the region based on the face's vertical center
        if face_center_y < upper_limit:
            color = red  # Upper region
            #position = "HIGH"
        elif face_center_y < middle_limit:
            color = yellow  # Middle region
            #position = "MIDDLE"
        else:
            color = green  # Lower region
            #position = "LOW"

        # Draw the rectangle with the chosen color
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        face_center_percentage = (face_center_y / height) * 100

        # Output the percentage of the y value
        print(f"{face_center_percentage:.2f}", flush=True)
        #print(face_center_y)

    # Display the resulting frame
    cv2.imshow('Head Tracking', frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
