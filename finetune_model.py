"""
from ultralytics import YOLO

# Load a model
model = YOLO("models/yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="utils_finetuning/data_visdrone.yaml", epochs=100, imgsz=640)
"""


from ultralytics import YOLO

# Load a model
model = YOLO('models/yolo11n_visdrone.pt')  # load an official model

PROJECT = '/home/ilan/Hackathon/Drone-Defense-Hackathon'  # project name
NAME = 'result_finetuing_VisDrone_D-Fire'  # run name

model.train(
   data = 'utils_finetuning/data_VisDrone_D-Fire.yaml',
   task = 'detect',
   epochs = 200,
   verbose = True,
   batch = 64,
   imgsz = 640,
   patience = 20,
   save = True,
   device = 0,
   workers = 8,
   project = PROJECT,
   name = NAME,
   cos_lr = True,
   lr0 = 0.0001,
   lrf = 0.00001,
   warmup_epochs = 3,
   warmup_bias_lr = 0.000001,
   optimizer = 'Adam',
   seed = 42,
)