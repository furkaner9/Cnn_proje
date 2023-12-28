import numpy as np
import cv2 
import os
import tensorflow as tf 
def load_data(data_path):
    data=[]
    labels =[]

    for class_name in os.listdir(data_path):
        class_path = os.path.join(data_path,class_name)
        for image_name in os.listdir(class_path):
            image_path =os.path.join(class_path,image_name)
            image=cv2.imread(image_path)
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            image=cv2.resize(image(224,224))
            data.append(image)
            labels.append(class_name)

    return np.array(data), np.array(labels)
from tensorflow import  keras 
from keras.models import Sequential 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))  # num_classes, marka sayısına eşittir

from sklearn.model_selection import train_test_split

data_path = "path/to/your/dataset"
data, labels = load_data(data_path)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))


