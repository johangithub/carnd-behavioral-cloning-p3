import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, Cropping2D, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.regularizers import l2
from time import time

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = 'data/IMG/' + filename
        imageBGR = cv2.imread(current_path)
        image = cv2.cvtColor(imageBGR, cv2.COLOR_BGR2RGB)
        images.append(image)

        # Multiple cameras
        if i == 0:
            st = float(line[3])
        elif i == 1:
            st = float(line[3]) + .12
        elif i == 2:
            st = float(line[3]) - .12
        measurements.append(st)

        # Flip images
        images.append(np.fliplr(image))
        measurements.append(-st)


X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(1, 1), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(100, activity_regularizer=l2(0.01)))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

callbacks = [
    EarlyStopping(monitor='val_loss', patience=2, verbose=1),
    ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, verbose=1),
    TensorBoard(log_dir='./logs/{}'.format(time()))
]

model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5, batch_size=128, callbacks=callbacks)

model.save('model.h5')
