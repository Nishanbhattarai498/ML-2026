import cv2


image =cv2.imread('openCv\image.jpg')

if image is not None:
    success=cv2.imwrite('openCv\saved_image.jpg', image)  # Save the image to a new file
    if success:
        print("Image saved successfully.")
    else:
        print("Error: Could not save the image.")
else:
    print("Error: Could not read the image.")
    