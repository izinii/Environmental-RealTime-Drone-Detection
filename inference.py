from ultralytics import YOLO

"""
# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)

# Run inference on an image
results = model("images/img2.jpg", save=True) # return a Result object
"""

model = YOLO("yolov8s.pt")
#print("\nModel info:", model.info(), "\n\n") # Display model information
results = model("images/fire_and_smoke/fire-8/valid/images/", save=True)