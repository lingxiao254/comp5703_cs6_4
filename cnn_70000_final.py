# -*- coding: utf-8 -*-
"""cnn_70000_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FibaLlaiSvDYOyFooX2CfSFoM6-HrS9K
"""

# Use GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

from google.colab import drive
drive.mount('/content/drive')

!mkdir -p drive
!google-drive-ocamlfuse -o nonempty drive

# Mount drive
from google.colab import drive
drive.mount('/content/drive')

import os
os.getcwd()
os.chdir('/content/drive/My Drive')
os.getcwd()

#import library
import pandas as pd
import numpy as np
from glob import glob
import fnmatch
import cv2
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import matplotlib.pylab as plt
import itertools
import sklearn
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
itertools
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
import keras
from keras import backend as K
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, model_from_json
from keras.optimizers import SGD, RMSprop, Adam, Adagrad, Adadelta
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, GlobalAveragePooling2D

X1=np.load('X_new.npy')
Y1=np.load('Y_new.npy')

X=X1[0:40000]
Y=Y1[0:40000]

def describe_data(x,y):
    print('Total number: {}'.format(len(x)))
    print('Number of IDC(-): {}'.format(np.sum(y==0)))
    print('Number of IDC(+): {}'.format(np.sum(y==1)))
    print('Percentage of IDC(+) : {:.2f}%'.format(100*np.mean(y)))
describe_data(X, Y)

x_sub = np.array(X)
x_sub=x_sub/255.0
x_sub_shape = x_sub.shape[1] * x_sub.shape[2] * x_sub.shape[3]
x_flat = x_sub.reshape(x_sub.shape[0], x_sub_shape)

r = pd.value_counts(Y)
print(r)
from imblearn.under_sampling import RandomUnderSampler
rus=RandomUnderSampler(ratio='auto')
x_flat_resample,y_sub_resample=rus.fit_sample(x_flat,Y)
len(x_flat_resample)
r = pd.value_counts(y_sub_resample)
print(r)

X_train, X_test, y_train, y_test = train_test_split(x_flat_resample, y_sub_resample, test_size=0.2, random_state = 2) # 0.2 test_size means 20%

Y_train_c = to_categorical(y_train, num_classes = 2)
Y_test_c = to_categorical(y_test, num_classes = 2)

for i in range(len(X_train)):
    height, width, channels = 50,50,3
    X_train_reshape = X_train.reshape(len(X_train),height,width,channels)
for i in range(len(X_test)):
    height, width, channels = 50,50,3
    X_test_reshape = X_test.reshape(len(X_test),height,width,channels)

df = pd.DataFrame()
df["labels"]=y_train
label = df['labels']

r = pd.value_counts(y_train)
print(r)

print(X_train_reshape.shape)
print(Y_train_c.shape)
print(X_test_reshape.shape)
print(Y_test_c.shape)

model = Sequential()
model.add(Conv2D(32,(3,3), activation='relu', input_shape=(50, 50, 3), padding = 'Same', strides = 2, ))
model.add(Conv2D(32,(3,3), padding = 'Same', activation='relu' ))
model.add(Conv2D(32,(3,3), padding = 'Same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(2, activation='softmax'))   
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

datagen = ImageDataGenerator(
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True)

class MetricsCheckpoint(Callback):
    def __init__(self, savepath):
        super(MetricsCheckpoint, self).__init__()
        self.savepath = savepath
        self.history = {}
    def on_epoch_end(self, epoch, logs=None):
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        np.save(self.savepath, self.history)

def plot_cm(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def plot_lc(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

history = model.fit_generator(datagen.flow(X_train_reshape,Y_train_c, batch_size=32),validation_data=(X_test_reshape,Y_test_c),
                        steps_per_epoch=len(X_train_reshape) / 32, epochs=100 ,callbacks = [MetricsCheckpoint('logs')],verbose=1)

score = model.evaluate(X_test_reshape,Y_test_c)
print(score)

y_pred = model.predict(X_test_reshape)
Y_pred_classes = np.argmax(y_pred,axis=1) 
Y_true = np.argmax(Y_test_c,axis=1)
dict_characters = {0: 'IDC(-)', 1: 'IDC(+)'}
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
plot_cm(confusion_mtx, classes = list(dict_characters.values())) 
plt.show()
plot_lc(history)
plt.show()
