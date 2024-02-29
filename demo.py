import tensorflow as tf
from tflite_runtime.interpreter import Interpreter
import cv2
import numpy as np
import time

# Load the TFLite model and allocate tensors.
interpreter = Interpreter(model_path="modelc9888basic.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up the USB camera
cap = cv2.VideoCapture(0)  # Change the camera index as needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)
cap.set(cv2.CAP_PROP_FPS,30)

pos=(20,60)
font=cv2.FONT_HERSHEY_SIMPLEX
height=1.5
weight=3
myColor=(255,0,0)

fps=0


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the input image
    input_data = cv2.resize(frame, (256, 256))
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float32)
    # Set the input tensor data.
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference.
    interpreter.invoke()

    # Get the output tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    #print(output_data[0].shape)
    #print("burdan once")
    # Process the output data as need# Assuming output_data[0] has shape (256, 256, 5)
    output_channels = np.argmax(output_data[0], axis=-1)

    # Create a color mapping for each class
    color_mapping = {
    0: [64, 42, 42],    # Road
    1: [255, 0, 0],     # Lane
    2: [128, 128, 96],  # Undrivable
    3: [0, 255, 102],   # Movable
    4: [204, 0, 255],   # Car
    }

    # Map the output channels to colors
    output_colored = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_idx, color in color_mapping.items():
        output_colored[output_channels == class_idx] = color

    # Display the resulting frame
    cv2.imshow('Combined Output', output_colored)
    #ed for your application.

    # Example: Print the predicted output for the first sample
    #print("Predicted Output for the first sample:")
    #print(output_data[0])

    # You can further process the output data based on your application requirements.

    # Display the resulting frame
    
    #cv2.imshow('frame', output_data[0])


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()