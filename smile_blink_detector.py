# import the necessary packages
import argparse
import cv2
import dlib
import imutils
import numpy as np
from imutils import face_utils
from imutils.video import FileVideoStream, VideoStream
import time
from scipy.spatial import distance as dist

# Initialize constant and variables
SAR_THRESH = 0.44
EYE_THRESH = 0.3
EYE_CONSEC_FRAMES = 3
counter = 0
r_counter = 0
l_counter = 0
total = 0
r_total = 0
l_total = 0

# function to calculate 'smile aspect ratio'
def smile_aspect_ratio(mouth, jaw):
    jaw_length = dist.euclidean(jaw[0], jaw[16])
    mouth_length = dist.euclidean(mouth[0], mouth[6])
    sar = mouth_length / jaw_length
    return sar

# function to calculate 'eye aspect ratio'
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
# ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detecto    r (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Extract facial landmarks for mouth, jaw, left and right eye
(m_start, m_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
(j_start, j_end) = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]
(l_start, l_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(r_start, r_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize the camera stream
print("[INFO] starting camera stream thread...")
vs = VideoStream(src=0).start()
time.sleep(1.0)

# loop over the frames from the video stream
while True:
    # grab the frame, resize it and convert to grayscale
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale frame
    faces = detector(gray, 0)

    # loop over the face detections
    for face in faces:
        # draw face bounding box
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # get the mouth and jaw values in the array and calculate the sar
        mouth_shape = shape[m_start:m_end]
        jaw_shape = shape[j_start:j_end]
        sar = smile_aspect_ratio(mouth_shape, jaw_shape)

        # drawing mouth contours
        mouth_hull = cv2.convexHull(mouth_shape)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

        # get the left and right eye values and calculate ear
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # drawing left and right eye contours
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # check sar and ear against respective threshold values
        if sar > SAR_THRESH:
            cv2.putText(
                            frame,
                            "Smiling",
                            (x + 60, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2,
                        )

        # Blink condition
        if ear < EYE_THRESH:
            counter += 1
        else:
            if counter >= EYE_CONSEC_FRAMES:
                total += 1
            counter = 0

        # left wink condition
        if EYE_THRESH > right_ear and left_ear < 0.93 * right_ear:
            l_counter += 1
        else:
            if l_counter >= EYE_CONSEC_FRAMES:
                l_total += 1
            l_counter = 0

        # right wink condition
        if EYE_THRESH > left_ear and right_ear < 0.93 * left_ear:
            r_counter += 1
        else:
            if r_counter >= EYE_CONSEC_FRAMES:
                r_total += 1
            r_counter = 0

        cv2.putText(
            frame,
            "Blinks: {} Left Wink: {} Right Wink: {}".format(total, l_total, r_total),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )


    # visualize the image
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # to close the image frame
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
