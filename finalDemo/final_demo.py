import os # operating system functions (ie. path building on Windows vs. MacOs)
import time # for time operations

from keras.models import load_model
from picamera.array import PiRGBArray
from picamera import PiCamera

from lib.hass import Hass
from lib.pseudo_concurrency import Timer

import yaml
import numpy as np # matrix operations (ie. difference between two matricies)
import cv2 # (OpenCV) computer vision functions (ie. tracking)

(MAJOR_VER, MINOR_VER, SUBMINOR_VER) = (cv2.__version__).split('.')
print('OpenCV Version: {}.{}.{}'.format(MAJOR_VER, MINOR_VER, SUBMINOR_VER))


# Helper function for applying a mask to an array
def mask_array(array, imask):
    if array.shape[:2] != imask.shape:
        raise Exception("Shapes of input and imask are incompatible")
    output = np.zeros_like(array, dtype=np.uint8)
    for i, row in enumerate(imask):
        output[i, row] = array[i, row]
    return output

def crop_image(image):
    return image[int(BBOX[1]):int(BBOX[1]+BBOX[3]), int(BBOX[0]):int(BBOX[0]+BBOX[2])]

## Constants

# Bounding box -> (TopRightX, TopRightY, Width, Height)
BBOX = (116, 116, 170, 170)

CLASSES = {

    0: 'fist',
    1: 'five',
    2: 'point',
    3: 'swing'
}

# path to model weights and architechture file
MODEL_FILE = os.path.join('lib', 'hand_model_gray.hdf5')
RES = (640, 480)

# Read and apply configuration, read model
with open('config.yml', 'r') as stream:
    CONFIG = yaml.load(stream)

HASS_ENDPOINT = '{https}://{host}:{port}/api/'.format(
    https='https' if CONFIG['home-assistant']['https'] else 'http',
    host=CONFIG['home-assistant']['host'],
    port=CONFIG['home-assistant']['port']
)


HASS = Hass(HASS_ENDPOINT, CONFIG['home-assistant']['access-token'])

LAMBDAS = {
    'fist': lambda: HASS.switch_toggle(CONFIG['switch-for-gesture']['fist']),
    'five': lambda: HASS.switch_toggle(CONFIG['switch-for-gesture']['five']),
    'point': lambda: HASS.switch_toggle(CONFIG['switch-for-gesture']['point']),
    'swing': lambda: HASS.switch_toggle(CONFIG['switch-for-gesture']['swing'])
}

GESTURE_TIMERS = {elem:Timer(LAMBDAS[elem], 5) for elem in CLASSES.values()}

# Kernel for erosion and dilation of masks
KERNEL = np.ones((3, 3), np.uint8)


# Display positions (pixel coordinates)
POSITIONS = {
    'hand_pose': (15, 40), # hand pose text
    'fps': (15, 20), # fps counter
    'null_pos': (200, 200) # used as null point for mouse control
}

HAND_MODEL = load_model(MODEL_FILE, compile=False)


## Variables and setup

# Camera setup
camera = PiCamera()
camera.resolution = RES
camera.framerate = 25
rawCapture = PiRGBArray(camera, size=RES)
time.sleep(0.5)

# Initialize background
camera.capture(rawCapture, format="bgr")
bg = crop_image(rawCapture.array)
rawCapture.truncate(0)
bg_timer = Timer(lambda: True, 1)

## Capture, process, display loop
for image in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # Read a new frame
    frame = image.array
    frame = cv2.flip(frame, 1)

    # Use numpy array indexing to crop the foreground frame
    cropped_frame = crop_image(frame)

    cropped_display = cropped_frame.copy()
    data_display = np.zeros_like(frame, dtype=np.uint8) # Black screen to display data

    # Get initial tick count
    tick_count = cv2.getTickCount()

    # Processing
    # First find the absolute difference between the two images
    diff = cv2.absdiff(bg, cropped_frame)
    mask = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    # Threshold the mask
    th, thresh = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
    # Opening, closing and dilation
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, KERNEL)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, KERNEL)
    img_dilation = cv2.dilate(closing, KERNEL, iterations=2)
    # Get mask indices
    imask = img_dilation > 0
    # Get foreground from mask
    foreground = mask_array(cropped_frame, imask)

    # Use numpy array indexing to crop the foreground frame
    hand_crop = img_dilation.copy()
    try:
        # Resize cropped hand and make prediction on gesture
        hand_crop_resized = np.expand_dims(
            cv2.resize(hand_crop, (54, 54)), axis=0
        ).reshape((1, 54, 54, 1))
        prediction = HAND_MODEL.predict(hand_crop_resized)
        predi = prediction[0].argmax() # Get the index of the greatest confidence
        gesture = CLASSES[predi]
        confidence = float(prediction[0][predi])
        if confidence > CONFIG['thresholds'][gesture]:
            GESTURE_TIMERS[gesture].execute_if_ready()

        for i, pred in enumerate(prediction[0]):
            # Draw confidence bar for each gesture
            barx = POSITIONS['hand_pose'][0]
            bary = 60 + i*60
            bar_height = 20
            bar_length = int(400 * pred) + barx # calculate length of confidence bar

            # Make the most confidence prediction green
            if i == predi:
                colour = (0, 255, 0)
            else:
                colour = (0, 0, 255)

            cv2.putText(
                data_display,
                "{}: {}".format(CLASSES[i], pred),
                (POSITIONS['hand_pose'][0], 30 + i*60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2
            )
            cv2.rectangle(
                data_display,
                (barx, bary),
                (bar_length, bary - bar_height),
                colour,
                -1,
                1
            )
    except Exception:
        pass

     # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_count)
    # Display FPS on frame
    cv2.putText(
        cropped_display,
        "FPS : " + str(int(fps)),
        POSITIONS['fps'],
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (50, 170, 50),
        2
    )
    # Display result
    cv2.imshow("cropped_display", cropped_display)
    # Display result
    cv2.imshow("data", data_display)
    k = cv2.waitKey(1) & 0xff

    if k == 27: # ESC pressed
        break
    if bg_timer.execute_if_ready():
        bg = crop_image(frame.copy())
    rawCapture.truncate(0)

cv2.destroyAllWindows()
camera.close()
