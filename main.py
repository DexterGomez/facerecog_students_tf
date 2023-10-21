import tensorflow as tf
import numpy as np
import cv2
import os

MODEL_DIR2 = './model_weigths/model_MbNV2_ep4_augT.h5'
MODEL_DIR1 = './model_weigths/model_Vgg16_ep4_augT.h5'

# Get persons names
data_dir = "./data/train/"
names = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
names = sorted(names)

# Load the trained model
#model = tf.keras.models.load_model('best_model_mnv2.h5')
model_01 = tf.keras.models.load_model(MODEL_DIR1)
model_02 = tf.keras.models.load_model(MODEL_DIR2)

# Start the webcam feed
cap = cv2.VideoCapture(1)

# You may need a face detector to detect multiple faces in a frame
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Get the face region
        face = frame[y:y+h, x:x+w]

        # Preprocess the face to input to the model
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        # Predict the person using both models
        predictions_01 = model_01.predict(face)
        predictions_02 = model_02.predict(face)

        # Combine the predictions (here, I'm averaging them)
        combined_predictions = (predictions_01 + predictions_02) / 2

        # Determine the predicted class and label
        predicted_class = np.argmax(combined_predictions, axis=1)[0]
        label = names[predicted_class] + ":" + str(round(np.max(combined_predictions, axis=1)[0], 2))

        print('predicted label: ' + label)
        print('avg pred model: ' + str(np.max(combined_predictions, axis=1)[0]))
        print('model1 pred: ' + str(np.max(predictions_01, axis=1)[0]))
        print('model2 pred: ' + str(np.max(predictions_02, axis=1)[0]))

        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame with bounding boxes and labels
    cv2.imshow('Video Feed', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release the capture
cap.release()
cv2.destroyAllWindows()