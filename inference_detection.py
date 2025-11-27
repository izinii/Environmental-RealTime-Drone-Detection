from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n_visdrone_D-Fire.pt")

# Display model information
#print("\nModel info:", model.info(), "\n\n") 

# Run inference
results = model("D-Fire_dataset/test/images/", save=True)