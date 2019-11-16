#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import os
import json
from datetime import datetime 

wrong_predictions = []
right_predictions = []

shape = 128
filter_size = 3


# In[2]:


def test(path, correct_category):
    test_image = image.load_img(path, target_size = (shape, shape))
    img = image.img_to_array(test_image)
    img /= 255
    img = np.expand_dims(img, axis = 0)
    result = classifier.predict(img)
    print(training_set.class_indices)
    for i in range(10):
        print(f"{result[0][i]:.4f}")
    predicted_index = np.argmax(result)
    was_correct = correct_category == predicted_index
    if was_correct:
        right_predictions.append(path)
    else:
        wrong_predictions.append(path)
    print(f"max val = {max(result[0])}, max index = {predicted_index}, correct category = {correct_category}, was correct = {was_correct}")    


# In[3]:


from datetime import datetime 

#time = datetime.now().strftime("%b-%d-%Y %H-%M-%S")

#model_filename = "model " + time + ".json"
#model_weights = "weights " + time + ".h5"

def serialize_model(model):
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")


# In[4]:


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = loaded_model.evaluate(X, Y, verbose=0)
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[5]:


def iterate(directory_key, directory):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(os.path.join(directory, filename))
            # test_image = image.load_img('dataset/Trainset/' + folder[i] + '/IMG_0845.jpg', target_size = (shape, shape))
            test((os.path.join(directory, filename).replace('\\', '/')), directory_key)
        else:
            continue


# In[6]:


# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(32, (filter_size, filter_size), input_shape = (shape, shape, 3), activation = 'relu'))
classifier.add(Conv2D(64, (filter_size, filter_size), activation = 'relu'))
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(128, (filter_size, filter_size), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Part 2 - Fitting the CNN to the images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
shear_range = 0.2,
zoom_range = 0.2,
horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('dataset/Trainset',
target_size = (shape, shape),
batch_size = 100,
class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('dataset/Testset',
target_size = (shape, shape),
batch_size = 100,
class_mode = 'categorical')


# In[ ]:


classifier.fit_generator(training_set,
#steps_per_epoch = 8000,
steps_per_epoch = 15,
epochs = 50,
validation_data = test_set,
validation_steps = 5)


# In[ ]:


# Part 3 - Making new predictions
import numpy as np
from tensorflow.keras.preprocessing import image
'''
test_image = image.load_img('dataset/Trainset/1_Haribo_Goldbaer/IMG_0715.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
print(result)
'''

train_test = {
    "1": "Trainset",
    "2": "Testset"
}

folder = {
    3: "1_Haribo_Goldbaer",
    4: "2_Chock_IT",
    5: "4_Kornflakes",
    #6: "5_K_Classic_Paprika_Chips",
    7: "6_Birnen_Dose",
    8: "7_Pfirsiche_Dose",
    9: "9_Aprikosen_Dose",
    0: "10_Choco_Haps",
    1: "11_K_Classic_Mehl",
    2: "12_K_Classic_Zucker"
}

for i in folder:
    iterate(i, 'dataset/real/' + folder[i])
#test_image = image.load_img('dataset/Trainset/' + folder[test_folder] + '/IMG_0845.jpg', target_size = (shape, shape))
#test_image_2 = image.load_img('dataset/Testset/'+ folder["2"] + '/IMG_3472.jpg', target_size = (shape, shape))
#test_image_3 = image.load_img('dataset/Trainset/ '+ folder["7"] + '/IMG_0551.jpg', target_size = (shape, shape))
print(f"right_predictions = {len(right_predictions)}, wrong_predictions = {len(wrong_predictions)}")
print(right_predictions)
print(wrong_predictions)
#test(test_image, test_folder)


# In[ ]:


serialize_model(classifier)


# In[ ]:


time = datetime.now().strftime("%b-%d-%Y %H-%M-%S")

filename = "predictions_test " + time + ".json"
data = {
    "wrong": wrong_predictions,
    "right": right_predictions
}

with open(filename, 'w') as outfile:
    json.dump(data, outfile)


# In[ ]:


import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(classifier.history['acc'])
plt.plot(classifier.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(classifier.history['loss'])
plt.plot(classifier.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

