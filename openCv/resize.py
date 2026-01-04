import cv2

image =cv2.imread('openCv\image.jpg')
if image is  None:
    print("Error: Could not read the image.")
else:
    print(f"Original Dimensions: {image.shape}")
    resized_image = cv2.resize(image, (400, 300))  # Resize the image to 400x300 pixels
    cv2.imshow('Resized Image', resized_image)  # Display the resized image in a window
    cv2.waitKey(0)  # Wait for a key press to close the window
    cv2.destroyAllWindows()  # Close all OpenCV windows

