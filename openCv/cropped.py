import cv2


image =cv2.imread('openCv\image.jpg')
if image is not None:

    cropped_image = image[50:200, 100:300]  # Crop the image (y1:y2, x1:x2)
    cv2.imshow('original Image', image)  # Display the original image in a window
    cv2.imshow('Cropped Image', cropped_image)  # Display the cropped image in a window
    cv2.waitKey(0)  # Wait for a key press to close the window  
    cv2.destroyAllWindows()  # Close all OpenCV windows
else:
    print("Error: Could not read the image.")   



