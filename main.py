import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
import cv2

mnist = tf.keras.datasets.mnist # 28x28 60000 training 10000 test
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = tf.keras.layers.Rescaling(scale=1./255, offset = 0.0)(x_train)
x_test = tf.keras.layers.Rescaling(scale=1./255, offset = 0.0)(x_test)

# change to np array
np_x_train = np.array(x_train)
np_x_test = np.array(x_test)

# increased the dimension for Convolution Kernel operations that follow
reshaped_x_train = np_x_train.reshape(-1, 28, 28, 1) # basically just envelopped this in another array/dimension
reshaped_x_test = np_x_test.reshape(-1, 28, 28, 1)
model = Sequential()

# First Convolution Layer
model.add(Conv2D(64, (3, 3), input_shape = reshaped_x_train.shape[1:])) # the input size only matters for the first conv layer
model.add(Activation("relu")) # removes values less than 0
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2))) ## Max Pooling Layer -> will only take the max value of a 2x2 matrix.
# 2nd Conv
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu")) # removes values less than 0
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2))) ## Max Pooling Layer -> will only take the max value of a 2x2 matrix.
# 3rd Conv
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu")) # removes values less than 0
model.add(Dropout(0.1))
model.add(MaxPooling2D(pool_size=(2,2))) ## Max Pooling Layer -> will only take the max value of a 2x2 matrix.

model.add(Flatten()) # need to flatten from 2D to 1D
### Fully Connected Layer  (each neuron is connected to all inputs)
model.add(Dense(64)) # 64 neurons, each is connected to ALL the inputs.
model.add(Activation("relu"))
model.add(Dropout(0.5))
### Fully Connected Layer 2 (each neuron is connected to all inputs)
model.add(Dense(32)) # 32 neurons, each is connected to ALL the inputs.
model.add(Activation("relu"))
model.add(Dropout(0.5))
# Add dropout layer here
## Last fully connected layer, output must be equal to a number so we have 10
model.add(Dense(10))
model.add(Activation('softmax')) # for probabilities
# before training, model needs to be compiled
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.fit(reshaped_x_train, y_train, epochs=5, validation_split = 0.3) # Training the model. Data, lavels, number of passes of the training data
test_loss, test_acc = model.evaluate(reshaped_x_test, y_test)
print("Test Loss on 10000 samples", test_loss)
print("Validation accuracy on 10000 samples", test_acc)

cap = cv2.VideoCapture("all2.mp4")
counter = 0
while True:
    _, frame = cap.read()
    counter += 1
    if counter % 2 == 0:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # resize
        resized = cv2.resize(gray_frame, (28, 28), interpolation = cv2.INTER_AREA) # check this last param
        # normalize
        normalized = tf.keras.utils.normalize(resized, axis=1)
        # reshape for the model
        reshaped = np.array(normalized).reshape(-1, 28, 28, 1)
        # make a prediction using model.predict()
        predictions  = model.predict(reshaped)
        number = np.argmax(predictions)
        cv2.rectangle(frame, (0, 0), (50, 50), (0, 0, 0), -1)
        cv2.putText(frame, number.astype(str), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        print("Prediction:", number)
        cv2.imshow("Handwritten Digits", frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

