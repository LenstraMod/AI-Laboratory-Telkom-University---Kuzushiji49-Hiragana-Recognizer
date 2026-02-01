import cv2
import numpy as np
from modules.image_utils import make_square
from modules.constants import k49_classmap

def predict_image(model,image_array, source="canvas"):
    if len(image_array.shape) > 2:
        if image_array.shape[2] == 4:
            img_gray = cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGBA2GRAY)   
        else:
            img_gray = cv2.cvtColor(image_array.astype("uint8"), cv2.COLOR_RGB2GRAY)
    else:
        img_gray = image_array.astype("uint8")
        
    if source == "upload":
        img_gray = make_square(img_gray)
    
    img_resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    
    img_inverted = cv2.bitwise_not(img_resized)
    img_normalized = img_inverted / 255.0
    
    img_final = np.expand_dims(img_normalized, axis=0)
    img_final = np.expand_dims(img_final, axis=-1)
    
    if model is not None:
        predict = model.predict(img_final)
        predicted_class = np.argmax(predict)
        chosen_char = k49_classmap[predicted_class]
        confidence = np.max(predict) * 100

    return chosen_char, confidence, img_inverted, predict