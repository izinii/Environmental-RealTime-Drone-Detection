import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import time

model_smoke = load_model("models/model_smoke_big.hdf5")
model_fire = load_model("models/model_fire_big.hdf5")
fire_threshold = 0
smoke_threshold = 0

image = cv2.imread("segmentation_data/photo_test.jpg")         
img_size = (image.shape[1],image.shape[0])
out = image.copy()

if not os.path.exists("/datas02/t0323469/hackathon/output_image_segmentation/"):
    os.mkdir("/datas02/t0323469/hackathon/output_image_segmentation/")
if not os.path.exists("/datas02/t0323469/hackathon/output_mask/"):
    os.mkdir("/datas02/t0323469/hackathon/output_mask/")

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
threshold_value = 0 # Choisir un seuil de 0.5 ou tout autre valeur
binary_mask = np.where(pred >= threshold_value, 255, 0).astype(np.uint8)  # Applique le seuil et convertit en 0 ou 255
cv2.imwrite("/datas02/t0323469/hackathon/output_mask/mask_fire_" + str(time.time()).replace(".", "") + ".png", binary_mask)
probability_mask = (pred * 255).astype(np.uint8)  # Multiplie par 255 et convertit en uint8
cv2.imwrite("/datas02/t0323469/hackathon/output_mask/probability_" + str(time.time()).replace(".", "") + ".png", probability_mask)
out[:, :, 1] = cv2.addWeighted(out[:, :, 1].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)

#smoke
pred = do_inference(image,model_smoke)
pred[pred<smoke_threshold]=0
threshold_value = 0  # Choisir un seuil de 0.5 ou tout autre valeur
binary_mask = np.where(pred >= threshold_value, 255, 0).astype(np.uint8)  # Applique le seuil et convertit en 0 ou 255
cv2.imwrite("/datas02/t0323469/hackathon/output_mask/mask_fire_" + str(time.time()).replace(".", "") + ".png", binary_mask)
probability_mask = (pred * 255).astype(np.uint8)  # Multiplie par 255 et convertit en uint8
cv2.imwrite("/datas02/t0323469/hackathon/output_mask/probability_" + str(time.time()).replace(".", "") + ".png", probability_mask)
out[:, :, 0] = cv2.addWeighted(out[:, :, 0].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)   

#Save the output images in a directory
cv2.imwrite("/datas02/t0323469/hackathon/output_image_segmentation/" + str(time.time()).replace(".","") + ".jpg",out)