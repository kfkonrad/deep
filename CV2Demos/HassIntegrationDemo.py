import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations

import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(major_ver, minor_ver, subminor_ver))

from keras.models import load_model

from picamera.array import PiRGBArray
from picamera import PiCamera
from hass import Hass

hass_token = open('auth.token', 'r').read()[:-1]
hass = Hass('http://localhost:8123/api/', hass_token)

MODEL_PATH = os.path.join('model')
MODEL_FILE = os.path.join(MODEL_PATH, 'hand_model_gray.hdf5') # path to model weights and architechture file
MODEL_HISTORY = os.path.join(MODEL_PATH, 'model_history.txt') # path to model training histor

hand_model = load_model(MODEL_FILE, compile=False)

# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def crop_image(image):
    return image[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]

classes = {
    0: 'fist',
    1: 'five',
    2: 'point',
    3: 'swing'
}

# Tracking
# Bounding box -> (TopRightX, TopRightY, Width, Height)
bbox_initial = (116, 116, 170,170)
bbox = bbox_initial

camera=PiCamera()
res = (640, 480)
camera.resolution = res
camera.framerate = 20
rawCapture = PiRGBArray(camera, size=res)
time.sleep(0.5)

camera.capture(rawCapture, format="bgr")
bg=crop_image(rawCapture.array)
rawCapture.truncate(0)

# Kernel for erosion and dilation of masks
kernel = np.ones((3,3),np.uint8)


# Display positions (pixel coordinates)
positions = {
    'hand_pose': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
    'null_pos': (200, 200) # used as null point for mouse control
}
gesture_timer = 0

# Capture, process, display loop
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Read a new frame
    frame = image.array
    frame = cv2.flip(frame, 1)

        
    # Use numpy array indexing to crop the foreground frame
    cropped_frame = frame[int(bbox[1]):int(bbox[1]+bbox[3]), int(bbox[0]):int(bbox[0]+bbox[2])]


    crop_display = cropped_frame.copy()
    data_display = np.zeros_like(frame, dtype=np.uint8) # Black screen to display data

    # Start timer
    tick_timer = cv2.getTickCount()

    # Processing
    # First find the absolute difference between the two images
    diff = cv2.absdiff(bg, cropped_frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img_dilation = cv2.dilate(closing, kernel, iterations=2)
    # Get mask indices
    imask = img_dilation > 0
    # Get foreground from mask
    foreground = mask_array(cropped_frame, imask)
        
    # Use numpy array indexing to crop the foreground frame
    hand_crop = img_dilation.copy()
    try:
        # Resize cropped hand and make prediction on gesture
        hand_crop_resized = np.expand_dims(cv2.resize(hand_crop, (54, 54)), axis=0).reshape((1, 54, 54, 1))
        prediction = hand_model.predict(hand_crop_resized)
        predi = prediction[0].argmax() # Get the index of the greatest confidence
        gesture = classes[predi]
        confidence = float(prediction[0][predi])
        current_time = time.time()
        if confidence > 0.9 and  current_time - gesture_timer > 5:
            if gesture == 'five':
                hass.switch_toggle('input_boolean.my_ip_a1')
                gesture_timer = current_time
        
        for i, pred in enumerate(prediction[0]):
            # Draw confidence bar for each gesture
            barx = positions['hand_pose'][0]
            bary = 60 + i*60
            bar_height = 20
            bar_length = int(400 * pred) + barx # calculate length of confidence bar
            
            # Make the most confidence prediction green
            if i == predi:
                colour = (0, 255, 0)
            else:
                colour = (0, 0, 255)
            
            cv2.putText(data_display, "{}: {}".format(classes[i], pred), (positions['hand_pose'][0], 30 + i*60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.rectangle(data_display, (barx, bary), (bar_length, bary - bar_height), colour, -1, 1)
    except Exception as ex:
        pass

     # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_timer)
    # Display FPS on frame
    cv2.putText(crop_display, "FPS : " + str(int(fps)), positions['fps'], cv2.FONT_HERSHEY_SIMPLEX, 0.65, (50, 170, 50), 2)
    
    # Display result
    cv2.imshow("crop_display", crop_display)
    # Display result
    cv2.imshow("data", data_display)
    k = cv2.waitKey(1) & 0xff
    
    if k == 27: break # ESC pressed
    elif k == 114 or k == 108: 
        # r pressed
        bg = crop_image(frame.copy())
        bbox = bbox_initial
    rawCapture.truncate(0)

cv2.destroyAllWindows()
camera.close()
