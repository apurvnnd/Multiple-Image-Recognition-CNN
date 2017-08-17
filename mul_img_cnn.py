#1 Building the CNN

#Importing the Keras Libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initializing the CNN
classifier = Sequential()

#Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation = 'relu'))

#Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size= (2, 2)))

# To increase Efficiency, add another Convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size= (2, 2)))

#Step 3 - Flattening
classifier.add(Flatten())

#Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
#Output layer
classifier.add(Dense(output_dim = 3, activation = 'softmax'))

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 : Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset1/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=5,
                                                 class_mode='categorical')
        

test_set = test_datagen.flow_from_directory('dataset1/test_set',
                                            target_size=(64, 64),
                                            batch_size=5,
                                            class_mode='categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch=(210/5),
                         epochs=20,
                         validation_data=test_set,
                         validation_steps=(90/5))


#Making New Predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset1/single_prediction/p1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
    prediction = 'person1'
elif result[0][1] == 1:
    prediction = 'person2'
elif result[0][2] == 1:
    prediction = 'person3'
else:
    prediction = 'none'
    
print(prediction)



