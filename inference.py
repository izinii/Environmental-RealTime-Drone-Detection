from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n_D-Fire.pt")

# Display model information
#print("\nModel info:", model.info(), "\n\n") 

# Run inference
results = model("images/fire_and_smoke/fire-8/test/images/", save=True)