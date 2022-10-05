import tensorflow as tf
from sklearn import model_selection
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, InputLayer
from keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape
from keras.utils.np_utils import to_categorical 
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# load data from encoded files
inputs = np.load("E:\DHS Robotics\demoCNN\ImagesProcessed\inputs.npy")
inputs = inputs.astype(np.float32)
labels = np.load("E:\DHS Robotics\demoCNN\ImagesProcessed\labels.npy")
labels = labels.astype(np.int32)

def CNNClassifier(num_epochs=2, layers=1, dropout=0.15):
  def create_model():
    model = Sequential()
    model.add(Reshape((128, 128, 3)))
    
    for i in range(layers):
      model.add(Conv2D(32, (3, 3), padding='same'))
      model.add(Activation('relu'))
    
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(4))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy'])
    return model
  return KerasClassifier(build_fn=create_model, epochs=num_epochs, batch_size=10, verbose=2)

# ploting accuracy over each iteration
def plot_acc(history, ax = None, xlabel = 'Epoch #'):
    history = history.history
    history.update({'epoch':list(range(len(history['val_accuracy'])))})
    history = pd.DataFrame.from_dict(history)

    best_epoch = history.sort_values(by = 'val_accuracy', ascending = False).iloc[0]['epoch']

    if not ax:
      f, ax = plt.subplots(1,1)
    sns.lineplot(x = 'epoch', y = 'val_accuracy', data = history, label = 'Validation', ax = ax)
    sns.lineplot(x = 'epoch', y = 'accuracy', data = history, label = 'Training', ax = ax)
    ax.axhline(0.5, linestyle = '--',color='red', label = 'Chance')
    ax.axvline(x = best_epoch, linestyle = '--', color = 'green', label = 'Best Epoch')  
    ax.legend(loc = 7)    
    ax.set_ylim([0.4, 1])

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Accuracy (Fraction)')
    
    plt.show()
    
# construct an image from the inputs for debugging
def display_image(data, labels, img_idx):
  import matplotlib.pyplot as plt
  my_img   = data[img_idx, :].squeeze().reshape([64,64,3]).copy()
  my_label = labels[img_idx]
  print('label: %s'%my_label)
  plt.imshow(my_img)
  plt.show()
  
# randomly divides data points into 80% for training and 20% for testing 
inputs_train, inputs_test, labels_train, labels_test = model_selection.train_test_split(inputs, labels, test_size=0.2, random_state=1)

# train CNN & plot progress
cnn = CNNClassifier(5, 1, 0)    # CNNClassifier(epochs, layers, dropout)
history = cnn.fit(inputs_train, labels_train, validation_data=(inputs_test, labels_test))
plot_acc(history)
preds = cnn.predict(inputs_test)
print (cnn.score(inputs_test, labels_test))

# save trained model to file
joblib.dump(cnn, "E:/DHS Robotics/demoCNN/ImagesProcessed/trained_model.pkl")