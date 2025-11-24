import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

model_smoke = load_model("models/model_smoke_big.hdf5")
model_fire = load_model("models/model_fire_big.hdf5")
fire_threshold = 0
smoke_threshold = 0

image = cv2.imread("segmentation_data/photo_test.jpg")         
img_size = (image.shape[1],image.shape[0])
out = image.copy()

def do_inference(image_,model):
    #preprocess the image and run the inference
    im = cv2.resize(image_, (896,512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255
    im2 = np.expand_dims(im, 0)
    preds = model.predict(im2)
    pred= preds[:, :, :, 0][0]
    return pred

#fire
pred = do_inference(image,model_fire)
pred[pred<fire_threshold]=0
out[:, :, 1] = cv2.addWeighted(out[:, :, 1].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)

#smoke
pred = do_inference(image,model_smoke)
pred[pred<smoke_threshold]=0
out[:, :, 0] = cv2.addWeighted(out[:, :, 0].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)   

#Save the output images in a directory
cv2.imwrite("test/" + "output_segmentation" + ".jpg", out)