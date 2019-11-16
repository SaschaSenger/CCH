# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid')) # units = 1 -> 10
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']) #binary_ -> categorical_

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('/home/nyxware/Trainset',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical') #binary -> categorical
test_set = test_datagen.flow_from_directory('/home/nyxware/Testset',
target_size = (64, 64),
batch_size = 32,
class_mode = 'categorical') # binary -> categorical
classifier.fit_generator(training_set,
steps_per_epoch = 30,
epochs = 5,
validation_data = test_set,
validation_steps = 20)

# Part 3 - Making new predictions
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/home/nyxware/Testset/6_Birnen_Dose/IMG_4346.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image /= 255
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)


for i in range(10):
    print(f"{result[0][i]:.2f}")
    
max_value_index = np.argmax(result)

for key, value in training_set.class_indices.items():
    if max_value_index == value:
        print("Prediction:" + key)
