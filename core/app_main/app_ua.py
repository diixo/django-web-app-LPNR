# https://www.kaggle.com/datasets/sarthakvajpayee/ai-indian-license-plate-recognition-data/code


import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import matplotlib.pyplot as plt
import numpy as np
import cv2
import tensorflow as tf

from sklearn.metrics import f1_score 
from keras import optimizers
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
from django.conf import settings


# detects and perfors blurring on the number plate.
def extract_plate(img, text=''):

    # Loads the data required for detecting the license plates from cascade classifier.
    plate_cascade = cv2.CascadeClassifier(os.path.join(settings.ML_ROOT, 'haarcascade_russian_plate_number.xml'))

    plate_img = img.copy()
    roi = img.copy()
    plate = None
    # detects numberplates and returns the coordinates and dimensions of detected license plate's contours.
    plate_rect = plate_cascade.detectMultiScale(plate_img, scaleFactor = 1.05, minNeighbors = 8)

    width_max = 0 # use for sorting by width
    plate_max = None
    x_max = 0
    y_max = 0

    for (x,y,w,h) in plate_rect:

        # perform proportional force sub-pixel zoom
        a, b = (int(0.1 * h), int(0.1 * w)) 
        aa, bb = (int(0.1 * h), int(0.1 * w))

        if h > 75: # skip width-splitting of high-quality image
            b = 0
            bb = 0

        plate = roi[y+a : y+h-aa, x+b : x+w-bb, :]

        if width_max < w:
            plate_max = plate
            width_max = w
            x_max = x
            y_max = y

        # finally representing the detected contours by drawing rectangles around the edges:
        cv2.rectangle(plate_img, (x+2,y), (x+w-3, y+h-5), (51,224,172), 3)
    if text != '':
        h = plate_max.shape[0]
        plate_img = cv2.putText(plate_img, text, (x_max, y_max-h//3), 
                                cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, (51,224,172), 2, cv2.LINE_AA)
        
    return plate_img, plate_max


# Testing the above function
def display(img_, title=''):
    img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)
    ax.imshow(img)
    plt.axis('off')
    plt.title(title)
    plt.show()


# Match contours to license plate or character template
def find_contours(dimensions, img):

    i_width_threshold = 6 #7

    # Find all contours in the image
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Retrieve potential dimensions
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    
    # Check largest 16 contours for license plate or character respectively
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:16]
    
    # input image binary plate: to convert img.shape(h,w) to img.shape(h,w,3)
    ii = np.dstack([img] * 3)
    
    x_cntr_list = []
    target_contours = []
    img_res = []
    for cntr in cntrs:
        # detects contour in binary image and returns the coordinates of rectangle enclosing it
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        
        # checking the dimensions of the contour to filter out the characters by contour's size
        if (intWidth >= i_width_threshold and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height) :
            x_cntr_list.append(intX) #stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((44, 24))
            # extracting each character using the enclosing rectangle's coordinates.
            char = img[intY:intY + intHeight, intX:intX + intWidth]

            if (intWidth >=i_width_threshold and intWidth < lower_width) :
                i_char = cv2.resize(char, (intWidth, 42), interpolation=cv2.INTER_LINEAR_EXACT)

                char = np.full((42, 22), 255, dtype=np.uint8)
                begin = int((22 - intWidth)/2) # center alignment
                char[:, begin:begin+intWidth] = i_char[:,:]
            else:
                char = cv2.resize(char, (22, 42), interpolation=cv2.INTER_LINEAR_EXACT)
            
            cv2.rectangle(ii, (intX, intY), (intWidth + intX, intY + intHeight), (50,21,200), 2)
            plt.imshow(ii, cmap='gray')

            # Make result formatted for classification: invert colors
            char = cv2.subtract(255, char)

            # Resize the image to 24x44 with black border
            char_copy[1:43, 1:23] = char
            char_copy[0:1, :] = 0
            char_copy[:, 0:1] = 0
            char_copy[43:44, :] = 0
            char_copy[:, 23:24] = 0

            img_res.append(char_copy) # List that stores the character's binary image (unsorted)
            if len(img_res) >= 10: break
            
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
    plt.title("rectangled extracted plate")
    plt.show()

    # arbitrary function that stores sorted list of character indeces
    indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
    img_res_copy = []
    for idx in indices:
        img_res_copy.append(img_res[idx])# stores character images according to their index
    img_res = np.array(img_res_copy)

    return img_res, ii


# Find characters in the resulting images
def segment_to_contours(image):

    new_height = 75 # set fixed height
    print("original plate[w,h]:", image.shape[1], image.shape[0], "new_shape: 333", new_height)

    # Preprocess cropped license plate image
    img_lp = cv2.resize(image, (333, new_height), interpolation=cv2.INTER_LINEAR_EXACT)


    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 112, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # img_binary_lp = cv2.erode(img_binary_lp, (3,3))
    # img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

    LP_WIDTH = img_binary_lp.shape[1]
    LP_HEIGHT = img_binary_lp.shape[0]

    # Make borders white
    img_binary_lp[0:1,:] = 255
    img_binary_lp[:,0:1] = 255
    img_binary_lp[new_height-1:new_height,:] = 255
    img_binary_lp[:,332:333] = 255

    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/24,
                LP_WIDTH/8,
                LP_HEIGHT/3,
                2*LP_HEIGHT/3]
    plt.imshow(img_binary_lp, cmap='gray')
    plt.title("original plate contour (binary)")
    plt.show()

    # Get contours within cropped license plate
    char_list, char_rects = find_contours(dimensions, img_binary_lp)
    return char_list, img_binary_lp, cv2.cvtColor(char_rects, cv2.COLOR_BGR2RGB) 

################################################################################

# Metrics for checking the model performance while training
def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)



class stop_training_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy', 0) > 0.9975):
            self.model.stop_training = True

callbacks = [stop_training_callback()]

################################################################################
# create and train model
def train_model(file_path: str):

    train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
    path = 'ai-indian-license-plate-recognition-data/data/data'

    train_generator = train_datagen.flow_from_directory(
            path + '/train',        # this is the target directory
            target_size=(28, 28),   # all images will be resized to 28x28
            batch_size=1,
            class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(
            path + '/val',          # this is the target directory
            target_size=(28, 28),   # all images will be resized to 28x28 batch_size=1,
            class_mode='categorical')

    model = Sequential()

    model.add(Conv2D(16, (24,24), input_shape=(28, 28, 3), activation='relu', padding='same'))

    # model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
    # model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(37, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=0.00005),
        metrics=['accuracy'])
        #metrics=[custom_f1score])

    model.summary()

    batch_size = 1
    model.fit(
          train_generator,
          steps_per_epoch = train_generator.samples // batch_size,
          validation_data = validation_generator,
          validation_steps = validation_generator.samples // batch_size,
          epochs = 80,
          verbose=1,
          callbacks=callbacks)

    model.save(filepath=file_path)
    return model
################################################################################

def load_model(file_path: str):
    return tf.keras.models.load_model(filepath=file_path)

################################################################################
def fix_dimension(img): 
    new_img = np.zeros((28,28,3))
    for i in range(3):
        new_img[:,:,i] = img
    return new_img

# Predicting the output string number by contours
def predict_result(ch_contours, model):
    dic = {}
    characters = '#0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for i,c in enumerate(characters):
        dic[i] = c

    output = []
    for i,ch in enumerate(ch_contours):

        img_ = cv2.resize(ch, (28,28)) # interpolation=cv2.INTER_LINEAR by default

        img = fix_dimension(img_)
        img = img.reshape(1, 28, 28, 3) #preparing image for the model

        y_ = np.argmax(model.predict(img, verbose=0), axis=-1)[0] #predicting the class
        character = dic[y_]
        output.append(character)
        
    plate_number = ''.join(output)
    return plate_number

################################################################################

def main():

    original = cv2.imread('ai-indian-license-plate-recognition-data/car-3-720.jpg')
    #original = cv2.imread('test/wv-1.jpg')
    #original = cv2.imread('test/lancia-beta-1978-test.jpg')
    #original = cv2.imread('test/2/audi-1.jpg')


    # Getting plate prom the processed image
    output_img, plate = extract_plate(original)

    # display processed image with plate-rectangle
    display(output_img, 'detected license plate in the input image')
    cv2.imwrite('build/1-output_img.jpg', output_img)

    # display plate-image with plate-rectangle
    display(plate, 'extracted license plate from the image')
    cv2.imwrite('build/2-plate.jpg', plate)

    # display segmented plate to contours
    chars, img_gray, char_rects = segment_to_contours(plate)
    cv2.imwrite('build/3-contour.jpg', img_gray)
    cv2.imwrite('build/4-rects.jpg', char_rects)

    for i in range(len(chars)):
        plt.subplot(1, 10, i+1)
        plt.imshow(chars[i], cmap='gray')
        plt.axis('off')


    # perform model loading before prediction
    file_path = "build/ua-license-plate-recognition-model-37v2.h5"

    #model = train_model(file_path)
    model = load_model(file_path)


    #show predicted string chars number
    predicted_str = predict_result(chars, model)
    predicted_str = str.replace(predicted_str, '#', '')
    print(predicted_str)

    # Segmented characters and their predicted value.
    plt.figure(figsize=(10,6))
    for i,ch in enumerate(chars):
        img = cv2.resize(ch, (28,28), interpolation=cv2.INTER_LINEAR_EXACT)
        #cv2.imwrite(str(i)+'.bmp', img)
        plt.subplot(3, 4, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'predicted: {predict_result(chars, model)[i]}')
        plt.axis('off')
    plt.show()

    ################################################################################

    if len(predicted_str) > 4:
        numbered_img, plate = extract_plate(original, predicted_str)
        display(numbered_img, 'recognized license plate number')
        cv2.imwrite('build/5-numbered.jpg', numbered_img)
