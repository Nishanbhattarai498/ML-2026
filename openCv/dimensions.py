import cv2


image =cv2.imread('openCv\image.jpg')
if image is not None:
    height, width, channels = image.shape
    print(f"Image Dimensions: Width={width}, Height={height}, Channels={channels}")
else:
    print("Error: Could not read the image.")
    