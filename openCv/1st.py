import cv2


image =cv2.imread('openCv\image.jpg')

if image is not None:
    cv2.imshow('Nishan', image)# Display the image in a window
    cv2.waitKey(0)# Wait for a key press to close the window
    cv2.destroyAllWindows()# Close all OpenCV windows

else:
    print("Error: Could not read the image.")
