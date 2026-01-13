
# Tiny-Yolo-v4-for-Blackberry-detection

This repository contains the files needed to run our Tiny YOLO v4 model which detects blackberries and separates them onto 3 separate classes based on ripeness.

- **Ripe**
- **Half-ripe**
- **Unripe**

This model was fine-tuned using google collab: https://colab.research.google.com/drive/1j_7PRMLn-3lQSYq0Nf1xSPJVjPzsONzE?usp=drive_link and the code is available in the link previously mentioned.

The model was trained using images captured in greenhouse environments, complemented with additional data to improve generalization.

## Repository Structure

```bash
Tiny-Yolo-v4-for-Blackberry-detection/
│
├── model/
│   ├── yolov4-tiny-custom.cfg      # Model configuration
│   ├── yolov4-tiny-custom_best.weights  # Trained weights
│   ├── obj.data                        # Data (only used for training)
│   ├── process.py                    # Training process
│   └── obj.names                # Class names
│
├── results/
│   ├── chart.png                    # Training curve
│   ├── detection.jpg              # Model inference on an image
│   └── detection2.jpg             # Image without inference
│
├── videos/
│   ├── color1.mp4                  # Sample videos
│   └── resultado.mp4
│
├── train/
│   └── .jpg, .png            # Training dataset
│ 
├── test/
│   └── .jpg, .png            # Testing dataset
│ 
├── val/
│   └── .jpg, .png            # Validation dataset
│
├── test.py                          # Script to use the model on an image
└── README.md
```
## Results
### Training curve
<img width="500" height="500" alt="Training curve" src="https://github.com/user-attachments/assets/ca2edc56-a330-4be9-b3fa-6593364520d7" />

### Demo image

![deteccion_resultado_3clases](https://github.com/user-attachments/assets/180b58ca-0071-48f3-b16b-ffae315eba4f)


## Testing the model
Using test.py, you should be able to test the model on any random picture you may want just by adjusting what the file specifies.

## Notes
- The model is optimized for speed and low computational cost, making it suitable for embedded and real-time systems.

- Performance may vary depending on lighting conditions, camera quality, and scene complexity.
