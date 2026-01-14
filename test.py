import cv2
import numpy as np

# PATHS 
NAMES_PATH = r"C:\Documentos Gil\Tareas Tec\Estancia Berrys\Dataset 3clases\obj.names"
CFG_PATH   = r"C:\Documentos Gil\Tareas Tec\Estancia Berrys\Dataset 3clases\yolov4-tiny-custom.cfg"
WEIGHTS_PATH = r"C:\Documentos Gil\Tareas Tec\Estancia Berrys\Dataset 3clases\yolov4-tiny-custom_best.weights"
IMAGE_PATH = r"C:\Documentos Gil\Tareas Tec\Estancia Berrys\Dataset1000\20221203_111012.jpg"   # image to test
OUTPUT_PATH = r"C:\Documentos Gil\Tareas Tec\Estancia Berrys\yolo_result.jpg"



CONF_THRESH = 0.80
NMS_THRESH = 0.4
# COLORS (BGR)
COLORS = {
    0: (0, 255, 0),  
    1: (0, 0, 255),   
    2: (255, 0, 0)    
}

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
    class_id = class_ids[i]
    color = COLORS[class_id]

    label = f"{classes[class_id]}: {confidences[i]:.2f}"

    cv2.rectangle(image, (x, y), (x + w_box, y + h_box), color, 2)
    cv2.putText(
        image, label, (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
    )


# SHOW RESULT
cv2.imwrite(OUTPUT_PATH,image)
cv2.waitKey(0)
cv2.destroyAllWindows()
