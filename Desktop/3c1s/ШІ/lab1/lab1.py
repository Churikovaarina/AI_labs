import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']
# species which we will classify,  0, 1, 2  - integers corresponding that classes
class_names = ['Setosa', 'Versicolor', 'Virginica']



def download():

    # downloads a file from a URL if it is not already in the cache.
    train_path = tf.keras.utils.get_file(TRAIN_URL.split('/')[-1], TRAIN_URL)
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)

    return train_path, test_path

def load_data(y_name='Species'):
    # returns the iris dataset as (train_x, train_y), (test_x, test_y)
    train_path, test_path = download()

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


(train_x, train_y), (test_x, test_y) = load_data()
print(train_x)

scaler = MinMaxScaler()
# noramlization
train_x = scaler.fit_transform(train_x)
# normalize data with knowledge from the train dataset
test_x = scaler.transform(test_x)

# define the model, this is linear stack of layers
model = Sequential()
# add layers
model.add(Dense(10, input_dim=train_x.shape[1], activation='relu'))  # Input layer
model.add(Dense(10, activation='relu'))  # Hidden layer
model.add(Dense(len(class_names), activation='softmax'))  # Output layer

# compile the model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=10, verbose=0)

# evaluate the model
loss, accuracy = model.evaluate(test_x, test_y, verbose=0)
print(f'Test Accuracy: {accuracy*100}')

print(f'Test loss: {loss*100}')

# plotting accuracy
plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()

# plotting loss
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()