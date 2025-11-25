import numpy as np
import cv2
from tensorflow.keras.models import load_model

class ZoneSegmentation:
    def __init__(self):
        self.model_fire = load_model("models/model_fire_big.hdf5")
        self.model_smoke = load_model("models/model_smoke_big.hdf5")
        self.fire_threshold = 0
        self.smoke_threshold = 0

    def do_inference(self, image_, model):
        """ preprocess the image and run the inference """
        im = cv2.resize(image_, (896,512))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = im/255
        im2 = np.expand_dims(im, 0)
        preds = model.predict(im2)
        pred= preds[:, :, :, 0][0]
        return pred

    def get_zones(self, image):
        """
        Params:
        img: NDArray 
            Input image as a numpy array.

        Returns:
        mask for the fire and mask for the smoke
        """
        img_size = (image.shape[1],image.shape[0])
        out = image.copy()

        #fire
        pred_fire = self.do_inference(image, self.model_fire)
        pred_fire[pred_fire<self.fire_threshold]=0
        #smoke
        pred_smoke = self.do_inference(image, self.model_smoke)
        pred_smoke[pred_smoke<self.smoke_threshold]=0
        
        return pred_fire, pred_smoke


if __name__ == '__main__':
    name = "DJI_0507_fired"
    image = cv2.imread("dataset_fire/" + name + ".png")    
    zone_detection = ZoneSegmentation()
    pred_fire, pred_smoke = zone_detection.get_zones(image)
    np.save("segmentation_pred/" + "fire_" + name + ".npy", pred_fire)
    np.save("segmentation_pred/" + "smoke_" + name + ".npy", pred_smoke)