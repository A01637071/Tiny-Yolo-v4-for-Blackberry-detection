import cv2
import numpy as np

# PATHS 
NAMES_PATH = "model/blackberry.names"
CFG_PATH   = "model/blackberry-tiny-yolov4.cfg"
WEIGHTS_PATH = "model/blackberry-tiny-yolov4.weights"
IMAGE_PATH = "test.jpg"   # image to test

CONF_THRESH = 0.5
NMS_THRESH = 0.4

# LOAD CLASSES
with open(NAMES_PATH, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# LOAD YOLO NETWORK
net = cv2.dnn.readNetFromDarknet(CFG_PATH, WEIGHTS_PATH)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# OUTPUT LAYERS
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# LOAD IMAGE
image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

h, w = image.shape[:2]

# CREATE BLOB & FORWARD PASS
blob = cv2.dnn.blobFromImage(
    image, 1/255.0, (416, 416), swapRB=True, crop=False
)
net.setInput(blob)
outputs = net.forward(output_layers)

# POST-PROCESSING
boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESH:
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            width = int(detection[2] * w)
            height = int(detection[3] * h)

            x = int(center_x - width / 2)
            y = int(center_y - height / 2)

            boxes.append([x, y, width, height])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NON-MAX SUPPRESSION
indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)

# DRAWING RESULTS
for i in indices.flatten():
    x, y, w_box, h_box = boxes[i]
    label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"

    cv2.rectangle(image, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
    cv2.putText(
        image, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
    )

# SHOW RESULT
cv2.imshow("YOLOv4-Tiny Inference", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
