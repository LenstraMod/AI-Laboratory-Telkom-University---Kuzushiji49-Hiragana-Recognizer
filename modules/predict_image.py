import cv2
from modules.auto_crop import auto_crop
import numpy as np
from modules.image_utils import make_square
from modules.constants import k49_classmap
import streamlit as st

def predict_image(model,image_array, source="canvas"):
    if len(image_array.shape) > 2:
        if image_array.shape[2] == 4:
            img_gray = cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGBA2GRAY)   
        else:
            img_gray = cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.astype("uint8")
     
     #Get image and make it square if uploaded   
    if source == "upload":
        # 1. strong denoise
        img_gray = cv2.fastNlMeansDenoising(img_gray, None, 15, 7, 21)

        # 2. detect ink via percentile
        thr = np.percentile(img_gray, 40)
        mask = img_gray < thr

        # 3. create clean image
        clean = np.ones_like(img_gray) * 255
        clean[mask] = img_gray[mask]

        img_gray = clean

        # 4. crop to content
        img_gray = auto_crop(img_gray)
        img_gray = make_square(img_gray)
        
    #Resize to 28x28
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    
    #Normalize pixel values to [0, 1]
    img_normalized = img_resized.astype("float32") / 255.0
    
    img_inverted = 1.0 - img_normalized
    
    temp = (img_inverted * 255).astype("uint8")
    temp = cv2.dilate(temp, np.ones((1,1), np.uint8))
    img_boostthick = temp.astype("float32") / 255.0
    
    # Shape: (1, 28, 28, 1)
    img_final = np.expand_dims(img_boostthick, axis=(0, -1))  

    img_inverted = (img_inverted * 255).astype("uint8")
    
    
    #predict using the model
    if model is not None:
        predict = model.predict(img_final)
        predicted_class = np.argmax(predict)
        chosen_char = k49_classmap[predicted_class]
        confidence = np.max(predict) * 100

    return chosen_char, confidence, img_inverted, predict