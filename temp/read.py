from __future__ import print_function
import os
import ast
import cv2

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras import optimizers
from keras.utils import np_utils
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Input, Dropout, Flatten, Conv2D, MaxPool2D, Activation, BatchNormalization
from keras.models import Sequential
from keras import backend as K

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from skimage.transform import resize

#print(os.listdir("data"))
row = 628
dim = (row,row)  # image dimension


# datapoint to image conversion
def draw_to_img(datapoints):
    images = []
    i = 0
    for data in datapoints:
        # stroke = ast.literal_eval(stroke)
        fig, ax = plt.subplots()

        for x, y in data:
            ax.invert_yaxis()
            ax.plot(x, y, linewidth=5,color= 'black')
            ax.axis('off')
        # render figure
        fig.canvas.draw()
       # plt.show()
        X = np.array(fig.canvas.renderer._renderer)
        plt.close("all")
        plt.clf()

        # resize, normalize and invert the image
        X = (resize(X, (row, row), order=1, clip=False,
               mode='constant', preserve_range=True)/ 255.)

        # channels
        X = np.logical_not(X[:, :, 0])*1
        #plt.imshow(X,cmap='gray')
        #plt.show()
        print('processed {}/{}'.format(i + 1, len(datapoints)), end = '\r', flush=True)
        i += 1
        plt.close(fig)
        images.append(X)

    print("\n")
    print ('Finished!')
    images = np.array(images)
    return images


# grayscale to rgb conversion
def to_rgb(img):
    img_rgb = np.asarray(np.dstack((img, img, img)), dtype=np.uint8)
    return img_rgb


# plot model results using this function
def plot_metrics_primary(acc, val_acc, loss, val_loss):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex='col', figsize=(20, 7))

    ax1.plot(acc, label='Train Accuracy')
    ax1.plot(val_acc, label='Validation accuracy')
    ax1.legend(loc='best')
    ax1.set_title('Accuracy')

    ax2.plot(loss, label='Train loss')
    ax2.plot(val_loss, label='Validation loss')
    ax2.legend(loc='best')
    ax2.set_title('Loss')
    plt.xlabel('Epochs')

TRAIN_PATH = "D:/data/"
TEST_PATH = "test_simplified.csv"
SUBMISSION_NAME = 'submission.csv'

train = pd.DataFrame()
for file in os.listdir(TRAIN_PATH):


    train_temp = (pd.read_csv(TRAIN_PATH + file, usecols=[1,3, 5], nrows=100))
    indx = (np.array(np.where(train_temp['recognized'].values==1)))
    indx = indx.flatten()
    train = train.append(train_temp.loc[indx])
    del train_temp,indx
   # print(train.shape)





# shuffle the training data
# train = shuffle(train, random_state=123)

# total number of classes
len(os.listdir(TRAIN_PATH))



# Model parameters
LEARNING_RATE = 0.001
N_CLASSES = train['word'].nunique() #number of classes
CHANNEL = 1
#
# print(N_CLASSES)
#
# fixing label in the training set
train['word'] = train['word'].replace(' ', '_', regex=True)
train['word'] = train['word'].replace('T', 't', regex=True)

# get labels and one-hot encode them.
classes_names = train['word'].unique()
labels = pd.get_dummies(train['word']).values
train.drop(['word'], axis=1, inplace=True)
train.drop(['recognized'], axis=1, inplace=True)
len(labels)

print(labels[100])


#  training datapoints stacked in a list
drawings_train = [ast.literal_eval(pts) for pts in train['drawing'].values]
len(drawings_train)


train_images = draw_to_img(drawings_train)

plt.imshow(train_images[0], cmap=plt.cm.binary_r)

np.save('train_images_2.npy',train_images)
np.save('labels_2.npy',labels)



#train_images = np.load('train_images.npy')
#labels = np.load('labels.npy')


test = pd.read_csv(TEST_PATH, usecols=[0, 2], nrows=None) # was 100 before
#  testing datapoints stacked in a list
drawings_test = [ast.literal_eval(pts) for pts in test['drawing'].values]
len(drawings_test)

test_images = draw_to_img(drawings_test)

train_images.shape, test_images.shape

x_train, x_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1, random_state=1)

x_train.shape, y_train.shape, x_val.shape, y_val.shape

img_train = train_images[0]
img_test = test_images[0]

plt.figure(figsize=(9, 3));
plt.subplot(1, 2, 1); plt.title('Train'); plt.axis('off');
plt.imshow(img_train, cmap='gray');
plt.subplot(1, 2, 2); plt.title('Test'); plt.axis('off');
plt.imshow(img_test, cmap='gray');

x_train.shape, y_train.shape, x_val.shape, y_val.shape




batch_size = 64
num_classes = N_CLASSES
epochs = 4
# dim = (200, 200) #image dimension
img_rows = row
img_cols = row
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_val.reshape(x_val.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
x_train = x_train.astype('float32')
x_val = x_val.astype('float32')

#x_train /= 255
#x_val /= 255

print('x_train shape:', x_train.shape)
print('x_val shape:', x_val.shape)

print(x_train.shape[0], 'train samples')
print(x_val.shape[0], 'test samples')


print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
print(x_train[5].shape)


print(input_shape)



def get_model():
    
    input_layer=Input(shape=(img_rows, img_cols, 1))
    
    x=Conv2D(filters=8,kernel_size=(5,5),padding='valid', activation='relu')(input_layer)
    x=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')(x)
    
    x=Conv2D(filters=16,kernel_size=(3,3),padding='valid', activation='relu')(x)
    x=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')(x)
    
    x=Conv2D(filters=32,kernel_size=(3,3),padding='valid', activation='relu')(x)
    x=MaxPool2D(pool_size=(2,2),strides=2,padding='valid')(x)
    
    
    x=Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu')(x)
    x=MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    
    #x=Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu')(x)
    #x=MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    
    #x=Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu')(x)
    #x=MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    
    #x=Conv2D(filters=64,kernel_size=(3,3),padding='same', activation='relu')(x)
    #x=MaxPool2D(pool_size=(2,2),strides=2,padding='same')(x)
    
    x=Flatten()(x)
    
    x=Dense(units=64)(x)
    x=Dense(units=N_CLASSES)(x) 
    
    output_layer=Activation('softmax')(x)
    model=Model(inputs=input_layer,outputs=output_layer)
    model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
    return model

model=get_model()
model.summary()


print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)

path_model='quickdraw_cnn.h5' 

#K.tensorflow_backend.clear_session() # destroys the current graph and builds a new one
model=get_model() 
#K.set_value(model.optimizer.lr,1e-3) # set the learning rate

h=model.fit(x=x_train,     
            y=y_train, 
            batch_size=64,
            epochs=1000,
            verbose=1, 
            validation_data=(x_val,y_val),
            shuffle=True,
            callbacks=[
                ModelCheckpoint(filepath=path_model),
            ]
            )

if K.image_data_format() == 'channels_first':
    test_images = test_images.reshape(test_images.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    
    
   
test_images_re = test_images.astype('float32')
#test_images_re /= 255

print(test_images_re.shape)


test_images.shape
#make predictions
predictions = model.predict(test_images_re)

x = np.argpartition(predictions[5], -3)[-3:]
print(x)


top_3_predictions = np.asarray([np.argpartition(pred, -3)[-3:] for pred in predictions])
top_3_predictions = ['%s %s %s' % (classes_names[pred[0]], classes_names[pred[1]], classes_names[pred[2]]) for pred in top_3_predictions]
test['word'] = top_3_predictions



len(top_3_predictions)
print(top_3_predictions[5])



submission = test[['key_id', 'word']]
submission.to_csv(SUBMISSION_NAME, index=False)
submission.head()