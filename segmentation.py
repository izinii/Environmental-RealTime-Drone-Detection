import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
import sys
import argparse
import time


parser = argparse.ArgumentParser(description="List of available arguments")
parser.add_argument('--video_file',
                        default=None, type=str,
                        help='Path to the source video file')
parser.add_argument('--fire', default=False, action="store_true",help='Enable the fire segmentation model')
parser.add_argument('--smoke', default=False, action="store_true",help='Enable the smoke segmentation model')
parser.add_argument('--fire_threshold', default=0.5, type=float,
                        help='Threshold applied to the fire segmentation output (value between 0 and 1)')
parser.add_argument('--smoke_threshold', default=0.5, type=float,
                        help='Threshold applied to the smoke segmentation output (value between 0 and 1)')
parser.add_argument('--display_images', default=False, action="store_true", help="Display the output of the algorithm")
parser.add_argument('--save_images', default=False, action="store_true",help='Save output images')
args = parser.parse_args()

#load selected models
if args.smoke:
    #model_smoke = load_model("../models/model_smoke_big.hdf5")
    model_smoke = load_model("feu_fumee/models/model_smoke.hdf5")
if args.fire:
    #model_fire = load_model("../models/model_fire_big.hdf5")
    model_fire = load_model("feu_fumee/models/model_fire.hdf5")
if not args.fire and not args.smoke:
    print("Please select at least one model (fire and/or smoke)")
    sys.exit(1)

if args.video_file is not None:
    video_file = args.video_file

if args.save_images is not None and not os.path.exists("/datas02/t0323469/hackathon/output_images/"):
    os.mkdir("/datas02/t0323469/hackathon/output_images/")


def do_inference(image_,model):
    #preprocess the image and run the inference
    im = cv2.resize(image_, (896//2,512//2))
    #im = cv2.resize(image_, (896,512))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = im/255
    im2 = np.expand_dims(im, 0)
    preds = model.predict(im2)
    pred= preds[:, :, :, 0][0]
    return pred

# Instantiate the grabber
if args.video_file is not None:
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print("Cannot open file " + video_file)
        sys.exit(1)
    video_fps = cap.get(cv2.CAP_PROP_FPS)


# ================================================
# NEW : Préparation du writer vidéo final
# ================================================
ret, first_frame = cap.read()
if not ret:
    print("Unable to read the first frame of the video.")
    sys.exit(1)

img_size = (first_frame.shape[1], first_frame.shape[0])
fourcc = cv2.VideoWriter_fourcc(*"mp4v")   # codec mp4
output_path = "/datas02/t0323469/hackathon/final_video_10s.mp4"
out_video = cv2.VideoWriter(output_path, fourcc, video_fps, img_size)

# Remettre la première frame dans le buffer
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
# ================================================

key = 0
ESC = 27

# Main loop
while key != ESC:
    status, image = cap.read()
    if not status:
        print("No Image")
        break           
    img_size = (image.shape[1],image.shape[0])
    out = image.copy()
    
    if args.fire:
        pred = do_inference(image,model_fire)
        #Blend the output mask with the input image
        pred[pred<args.fire_threshold]=0
        out[:, :, 1] = cv2.addWeighted(out[:, :, 1].astype('uint8'), 1.0,
                                       cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)
    if args.smoke:
        pred = do_inference(image,model_smoke)
        #Blend the output mask with the input image
        pred[pred<args.smoke_threshold]=0
        out[:, :, 0] = cv2.addWeighted(out[:, :, 0].astype('uint8'), 1.0,
                                       cv2.resize((pred * 255).astype('uint8'), img_size).astype('uint8'), 1.0, 0)   

    # ================================================
    # NEW: Ajouter la frame segmentée à la vidéo finale
    # ================================================
    out_video.write(out)
    # ================================================

    if args.save_images:
        #Save the output images in a directory
        cv2.imwrite("/datas02/t0323469/hackathon/output_images/" + str(time.time()).replace(".","") + ".jpg",out)
        cv2.waitKey(1)
    if args.display_images:
        #Display the output images on screen
        cv2.imshow("output",out)
        cv2.waitKey(1)


# ================================================
# NEW: Terminer proprement la vidéo de sortie
# ================================================
out_video.release()
print("✔ Video saved to:", output_path)