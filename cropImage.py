from PIL import Image
import os

# Constants 
IDEALWIDTH, IDEALHEIGHT = 128, 128
DIR = 'E:\DHS Robotics\demoCNN\ImagesData'
RESULTS = 'E:\DHS Robotics\demoCNN\ImagesProcessed'

if not os.path.exists(RESULTS):
    os.mkdir(RESULTS)
    
def imageCrop(imageName, subdirectory):
    im = Image.open(DIR + "/" + subdirectory + "/" + imageName)
    width, height = im.size
    
    im1 = im.crop(((width - height) / 2, 0, width - (width - height) /2, height))
    im1 = im1.resize((IDEALWIDTH, IDEALHEIGHT))
    im1 = im1.save(f"" + RESULTS + "/" + subdirectory + "/" + imageName)
    
# loop through all folders in DIR
for subdirectory in os.listdir(DIR):
    if os.path.isdir(DIR + "/" + subdirectory) and subdirectory != "_results":
        # create mirror folder in result folder
        if not os.path.exists(RESULTS + "/" + subdirectory):
            os.mkdir(RESULTS + "/" + subdirectory)
        
        # loop through all images in folder
        for image in os.listdir(DIR + "/" + subdirectory):
            if os.path.isfile(DIR + "/" + subdirectory + "/" + image):
                imageCrop(image, subdirectory)