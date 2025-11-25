


class VLMevaluation:
    def __init__(self):
        self.model_fire = load_model("models/model_fire_big.hdf5")
        self.model_smoke = load_model("models/model_smoke_big.hdf5")
        self.fire_threshold = 0
        self.smoke_threshold = 0

    def evaluation(self, image):
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
        
        print(pred_fire)
        print("\n\n")
        print(pred_smoke)
        return pred_fire, pred_smoke


image = cv2.imread("03_0002.png")   
vlm_eval = VLMevaluation()
vlm_eval.evaluation(image)