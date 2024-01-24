import os
import keras 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('dark_background')
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder()
encoder.fit([[0], [1]]) 

# 0 for Tumor
# 1 for Normal

# with tumor

data = []       # for storing images data into numpy array form 
paths = []      # stores the path of all the images
result = []     # stores one hot encoded data

for r, d, f in os.walk(r'path to MRI images having tumor'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[0]]).toarray())

# without tumor

paths = []
for r, d, f in os.walk(r'path to tumor free MRI images'):
    for file in f:
        if '.jpg' in file:
            paths.append(os.path.join(r, file))

for path in paths:
    img = Image.open(path)
    img = img.resize((128,128))
    img = np.array(img)
    if(img.shape == (128,128,3)):
        data.append(np.array(img))
        result.append(encoder.transform([[1]]).toarray())

data = np.array(data)
result = np.array(result)
result = result.reshape(-1,2)

x_train,x_test,y_train,y_test = train_test_split(data, result, test_size=0.2, shuffle=True, random_state=0)

model = Sequential()

model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(128, 128, 3), padding = 'Same'))
model.add(Conv2D(32, kernel_size=(2, 2),  activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))
model.add(Conv2D(64, kernel_size = (2,2), activation ='relu', padding = 'Same'))

model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())        # 1D array me convert

model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss = "categorical_crossentropy", optimizer='Adamax',metrics=['accuracy'])
print(model.summary())

y_train.shape
#fitting the data into the model 
history = model.fit(x_train, y_train, epochs = 30, batch_size = 30, verbose = 1,validation_data = (x_test, y_test))

evaluation_result = model.evaluate(x_test, y_test, verbose=0)
test_loss = evaluation_result[0]
test_accuracy = evaluation_result[1]
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.legend(['Test', 'Validation'], loc='upper right')
plt.show()

def names(number):
    if number==0:
        return 'Its a Tumor'
    else:
        return 'No, Its not a tumor'

model.save('brain_tumor_detection_model.h5')

# Save model architecture to JSON file
model_json = model.to_json()
with open('brain_tumor_detection_model.json', 'w') as json_file:
    json_file.write(model_json)

# Save model weights to HDF5 file
model.save_weights('brain_tumor_detection_model_weights.h5')

