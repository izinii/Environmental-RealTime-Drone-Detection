import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import time

fire_threshold = 0
smoke_threshold = 0

model_smoke = load_model("feu_fumee/models/model_smoke_big.hdf5")
model_fire = load_model("feu_fumee/models/model_fire_big.hdf5")

if not os.path.exists("/datas02/t0323469/hackathon/output_test/"):
    os.mkdir("/datas02/t0323469/hackathon/output_test/")

def do_inference(image_,model):
    #preprocess the image and run the inference
    im = cv2.resize(image_, (896,512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255
    im2 = np.expand_dims(im, 0)
    preds = model.predict(im2)
    pred= preds[:, :, :, 0][0]
    return pred

image = cv2.imread("photo_test.jpg")         
img_size = (image.shape[1],image.shape[0])
out = image.copy()

#fire
pred = do_inference(image,model_fire)
pred[pred<fire_threshold]=0

threshold_value = 0 # Choisir un seuil de 0.5 ou tout autre valeur
binary_mask = np.where(pred >= threshold_value, 255, 0).astype(np.uint8)  # Applique le seuil et convertit en 0 ou 255
cv2.imwrite("/datas02/t0323469/hackathon/test/mask_fire_" + str(time.time()).replace(".", "") + ".png", binary_mask)
probability_mask = (pred * 255).astype(np.uint8)  # Multiplie par 255 et convertit en uint8
cv2.imwrite("/datas02/t0323469/hackathon/test/probability_" + str(time.time()).replace(".", "") + ".png", probability_mask)

out[:, :, 1] = cv2.addWeighted(out[:, :, 1].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)
#smoke
pred = do_inference(image,model_smoke)
pred[pred<smoke_threshold]=0
print(pred)
threshold_value = 0  # Choisir un seuil de 0.5 ou tout autre valeur
binary_mask = np.where(pred >= threshold_value, 255, 0).astype(np.uint8)  # Applique le seuil et convertit en 0 ou 255
cv2.imwrite("/datas02/t0323469/hackathon/try/mask_fire_" + str(time.time()).replace(".", "") + ".png", binary_mask)
probability_mask = (pred * 255).astype(np.uint8)  # Multiplie par 255 et convertit en uint8
cv2.imwrite("/datas02/t0323469/hackathon/try/probability_" + str(time.time()).replace(".", "") + ".png", probability_mask)

out[:, :, 0] = cv2.addWeighted(out[:, :, 0].astype('uint8'), 1.0,
                            cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)   

#Save the output images in a directory
cv2.imwrite("/datas02/t0323469/hackathon/output_test/" + str(time.time()).replace(".","") + ".jpg",out)