import cv2

#AI Generated for this function
def make_square(image):
    old_size = image.shape[:2] # (Tinggi, Lebar)
    desired_size = max(old_size)
    
    # Hitung berapa hitam-hitam yang harus ditambahkan
    delta_w = desired_size - old_size[1]
    delta_h = desired_size - old_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Tambahkan bingkai hitam (Padding)
    # PENTING: Value=[255,255,255] (Putih) karena kita asumsikan inputnya kertas putih
    # Nanti baru di-invert jadi hitam.
    color = [255, 255, 255] 
    new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_im