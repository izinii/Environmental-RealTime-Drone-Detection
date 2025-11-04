from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Display model information
model.info()

# Train the model
#results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640)

# Run inference on an image
#results = model("images/img3.jpg", show=True, save=True) # return a Result object