import cv2
import numpy as np

capture = cv2.VideoCapture(0)

# weight and height
wht = 320
# threshold confidence
CONFIG_THRESOLD = 0.5
NMS_THRESHOLD = 0.3

# open coco file
class_path = 'coco.names'
class_names = []
with open(class_path, 'rt') as f:
    class_names = f.read().rstrip('\n').split('\n')
# print(class_names)

# load the config file and weights
model_config = 'yolov3.cfg'
model_weights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(model_config, model_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)


# function to draw bounding box

def find_objects(outputs, img):
    height, weight, channel = img.shape
    bbox = []
    class_ids = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > CONFIG_THRESOLD:
                w, h = int(det[2] * weight), int(det[3] * height)
                x, y = int((det[0] * weight) - w / 2), int((det[1] * height) - h / 2)
                bbox.append([x, y, w, h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, CONFIG_THRESOLD, NMS_THRESHOLD)
    for i in indices:
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{class_names[class_ids[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)


while True:
    success, img = capture.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (wht, wht), [0, 0, 0], 1, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_names = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)

    find_objects(outputs, img)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)

    # to close the image frame
    if key == ord("q"):
        break

cv2.destroyAllWindows()
