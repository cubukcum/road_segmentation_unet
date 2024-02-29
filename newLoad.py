import cv2
import numpy as np
import tensorflow as tf
import time
import sys

def run(model_path: str, display_mode: str, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:

    # Load the TensorFlow Lite model.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Variables to calculate FPS
    counter, fps = 0, 0
    start_time = time.time()

    # Start capturing video input from the camera
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Continuously capture images from the camera and run inference.
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            sys.exit('ERROR: Unable to read from webcam. Please verify your webcam settings.')

        counter += 1
        image = cv2.flip(image, 1)  # MAY NOT NEED IF USING A USB CAMERA

        # Convert the image from BGR to RGB as required by the TFLite model.
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_image = rgb_image / 255.0  # Normalize the image

        # Resize the image to match the model's expected sizing
        input_shape = input_details[0]['shape'][1:3]
        resized_image = cv2.resize(rgb_image, tuple(input_shape))

        # Quantize the input image
        input_data = resized_image.astype(np.uint8)

        # Set the input tensor
        interpreter.set_tensor(input_details[0]['index'], input_data)

        # Run inference
        interpreter.invoke()

        # Get the output tensor
        segmentation_result = interpreter.get_tensor(output_details[0]['index'])

        # Post-process the segmentation result (assuming segmentation_result is quantized)
        segmentation_mask = np.argmax(segmentation_result, axis=-1).astype(np.uint8)

        # Visualize segmentation result on image
        overlay = visualize(image, segmentation_mask, display_mode, fps)

        # Calculate the FPS
        if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
            end_time = time.time()
            fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
            start_time = time.time()

        # Stop the program if the ESC key is pressed.
        if cv2.waitKey(1) == 27:
            break
        cv2.imshow('image_segmentation', overlay)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    model_path = 'path/to/your/quantized/model.tflite'
    display_mode = 'overlay'  # or 'side-by-side' based on your preference
    num_threads = 4  # adjust as needed
    enable_edgetpu = False  # adjust as needed
    camera_id = 0  # adjust as needed
    width = 256  # adjust as needed
    height = 256  # adjust as needed

    run(model_path, display_mode, num_threads, enable_edgetpu, camera_id, width, height)
