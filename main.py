import cv2
import numpy as np

# Open the video file
cap = cv2.VideoCapture('video.mp4')

# Minimum dimensions for a valid vehicle detection
min_width_react = 80
min_height_react = 80
# Position of the line for counting vehicles
count_line_position = 550

# Initialize the background subtractor
algo = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x, y, w, h):
    # Calculate the center of the bounding rectangle.
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

# List to store detected vehicle centers
detect = []
offset = 6  # Offset for line detection
counter = 0  # Vehicle counter

while True:
    ret, frame1 = cap.read()

    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)

    # Apply background subtraction and morphological operations
    img_sub = algo.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contourShape, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame1, (25, count_line_position), (1200, count_line_position), (255, 127, 0), 3)

    for (i, c) in enumerate(contourShape):
        (x, y, w, h) = cv2.boundingRect(c)
        if (w >= min_width_react) and (h >= min_height_react):
            # Draw bounding box and label
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame1, "Vehicle " + str(counter), (x, y-20), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 244, 0), 2) 
            center = center_handle(x, y, w, h)
            detect.append(center)
            cv2.circle(frame1, center, 4, (0, 0, 255), -1)  

            # Check if any detected center crosses the counting line
            for (x, y) in detect:
                if y < (count_line_position + offset) and y > (count_line_position - offset):
                    counter += 1
                    detect.remove((x, y))
                    print("Vehicle Counter:", str(counter))
                    
    # Display the vehicle count
    cv2.putText(frame1, "VEHICLE COUNT: " + str(counter), (350, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (34, 34, 178), 5)            

    # Show the frame with annotations
    cv2.imshow('Video Original', frame1)
    
    # Exit if 'Enter' key is pressed
    if cv2.waitKey(1) == 13:
        break

# Release resources and close all windows
cv2.destroyAllWindows()
cap.release()