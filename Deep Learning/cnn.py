# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

import os
os.chdir('C:\DeepLearning\P16-Convolutional-Neural-Networks\Convolutional_Neural_Networks\dataset')

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution

# nb_filter = the number of feature maps we create (nb feature detectors)
# (3,3) --> feature detectors shape
# input_shape --> shape we expect
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))



# Step 2 - Pooling

# reduce the time we spent without losing features ( reduce the size of feature maps )
classifier.add(MaxPooling2D(pool_size = (2, 2))) # in general

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
# Q: why we don't lose spatial structure of feature maps
# A: because we use convolution step to extract the features through feature detector and
# we use max-pooling to extract the max of the feature map, which keeps the spatial features.
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')


test_set = test_datagen.flow_from_directory('test_set',
                                            target_size = (64, 64), # if we want to increase the accuracy, we need to increase the size so that we get more info from the image.
                                            batch_size = 32, # 每次训练32个样本
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000//32,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000//32)


import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single_prediction/cat_or_dog_23.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)

# add a new dimension
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)

#####################################
# IMPROVED
#####################################

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Image dimensions
img_width, img_height = 150, 150 # we use (64,64) previously. I think that is why the convergence is slow

"""
    Creates a CNN model
    p: Dropout rate
    input_shape: Shape of input 
"""


def create_model(p, input_shape=(32, 32, 3)):
    # Initialising the CNN
    model = Sequential()
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution + Pooling Layer
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # we have 4 convolution + pooling layers in total compared to the original one (only 2)

    # Flattening
    model.add(Flatten())
    # Fully connection
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p)) # avoid overfitting (train accuracy is high, but test accuracy is low)
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(p / 2))
    model.add(Dense(1, activation='sigmoid'))

    # Compiling the CNN
    optimizer = Adam(lr=1e-3)
    metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics)
    return model


"""
    Fitting the CNN to the images.
"""


def run_training(bs=32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1. / 255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory('training_set',
                                                     target_size=(img_width, img_height),
                                                     batch_size=bs,
                                                     class_mode='binary')

    test_set = test_datagen.flow_from_directory('test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='binary')

    model = create_model(p=0.6, input_shape=(img_width, img_height, 3))
    model.fit_generator(training_set,
                        steps_per_epoch=8000 / bs,
                        epochs=epochs,
                        validation_data=test_set,
                        validation_steps=2000 / bs)


def main():
    run_training(bs=32, epochs=100)


""" Main """
if __name__ == "__main__":
    main()
