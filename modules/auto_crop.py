import cv2

#this function made by AI
def auto_crop(img):
    mask = img < 200        # dark strokes
    coords = cv2.findNonZero(mask.astype("uint8"))
    if coords is not None:
        x,y,w,h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]
    return img