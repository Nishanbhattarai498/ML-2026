import cv2


image =cv2.imread('openCv\image.jpg')
if image is not None:
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert the image to grayscale
    cv2.imshow('Grayscale Image', gray)  # Display the grayscale image in a
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close all OpenCV windows
else:
    print("Error: Could not read the image.")
        