from PIL import Image
import os
import numpy as np

# constants
DIR = 'E:\DHS Robotics\demoCNN\ImagesProcessed'
IMAGE_SIZE = 128 * 128 * 3
DATA_SIZE = 1200 * 4
ONE_HOT_ENCODING = ["Symbol 1", "Symbol 2", "Symbol 3", "EMPTY"]

# set up data structures
inputs = np.empty((0, IMAGE_SIZE))
labels = np.empty((0, ONE_HOT_ENCODING.__len__()))

index = 0

for image in os.listdir(DIR + "/1"):
    image = Image.open(DIR + "/1/" + image)
    data = np.asarray(image)
    data = data.flatten()
    
    inputs = np.append(inputs, np.array([data]), axis = 0)
    labels = np.append(labels, np.array([[1, 0, 0, 0]]), axis = 0)    
    
    index += 1

for image in os.listdir(DIR + "/2"):
    image = Image.open(DIR + "/2/" + image)
    data = np.asarray(image)
    data = data.flatten()
    
    inputs = np.append(inputs, np.array([data]), axis = 0)
    labels = np.append(labels, np.array([[0, 1, 0, 0]]), axis = 0)    
    
    index += 1

for image in os.listdir(DIR + "/3"):
    image = Image.open(DIR + "/3/" + image)
    data = np.asarray(image)
    data = data.flatten()
    
    inputs = np.append(inputs, np.array([data]), axis = 0)
    labels = np.append(labels, np.array([[0, 0, 1, 0]]), axis = 0)    
    
    index += 1
    
for image in os.listdir(DIR + "/4"):
    image = Image.open(DIR + "/4/" + image)
    data = np.asarray(image)
    data = data.flatten()
    
    inputs = np.append(inputs, np.array([data]), axis = 0)
    labels = np.append(labels, np.array([[0, 0, 0, 1]]), axis = 0)    
    
    index += 1

print(inputs.shape)
print(labels.shape)
np.save(DIR + "/inputs.npy", inputs)
np.save(DIR + "/labels.npy", labels)