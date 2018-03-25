import pandas as pd 
import numpy as np
import os 
import keras
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Flatten
from keras.optimizers import Adam
from PIL import Image
from scipy import misc
from sklearn import preprocessing

class Model:
    # input shape
    def __init__(self, load=''):
        self.input_shape = (90,120,1)

        if load != '':
            print('Init model load: {0:}'.format(load))
            self.model = load_model(load)
        else:
            self.model = Sequential()
            self.model.add(Conv2D(20, kernel_size=(7, 9), strides=(1, 1), activation='relu', input_shape=self.input_shape))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Conv2D(50, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Conv2D(70, kernel_size=(4, 5), strides=(1, 1), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
            self.model.add(Activation('relu'))
            self.model.add(Flatten())
            self.model.add(Dense(500))
            self.model.add(Activation('relu'))
            self.model.add(Dense(5))
            self.model.add(Activation('softmax'))

    def GrayItAll(self,path,csvName):
        if (os.path.isdir(path + "/Images")) and (os.path.exists(path + "/" + csvName + ".csv")):
            df = pd.read_csv(path + "/" + csvName + ".csv")
            for index, row in df.iterrows():
                path = df.at[index,'fileName']
                # converts to grayScale
                img = Image.open(path).convert('L')
                img.save(path)

    def LoadTrainingData(self,path,csvName):
        if (os.path.isdir(path + "/Images")) and (os.path.exists(path + "/" + csvName + ".csv")):
            
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.,1.))

            df = pd.read_csv(path + "/" + csvName + ".csv")
            y_df = df[['action']]
            y_data = np.array(y_df)
            y_list = []
            # FD BK RT LT NO
            for element in y_data:
                if element == ['FD']:
                    y_list.append([1.,0.,0.,0.,0.])
                    continue
                if element == ['BK']:
                    y_list.append([0.,1.,0.,0.,0.])
                    continue
                if element == ['RT']:
                    y_list.append([0.,0.,1.,0.,0.])
                    continue
                if element == ['LT']:
                    y_list.append([0.,0.,0.,1.,0.])
                    continue
                if element == ['NO']:
                    y_list.append([0.,0.,0.,0.,1.])
                    continue
            self.y_data = np.array(y_list)
                
            # load images
            X_data = []
            for index, row in df.iterrows():
                path = df.at[index,'fileName']
                # reads image 
                img = misc.imread(path)
                img = min_max_scaler.fit_transform(img)
                X_data.append(img)
                # converts to grayScale
                # img = Image.open(path).convert('L')
                # img.save(path)
            arr = np.array(X_data)
            self.X_data = arr.reshape(arr.shape[0],arr.shape[1],arr.shape[2],1)
            print('X_data Shape: {0:} dtype: {1:}'.format(self.X_data.shape, self.X_data.dtype))
            print('y_data Shape: {0:} dtype: {1:}'.format(self.y_data.shape, self.y_data.dtype))

    def CompileAndFit(self,EPOCHS=400, INIT_LR=0.0001, BS=2,fileName='default.h5'):
        #opt = Adam(lr=INIT_LR, decay=INIT_LR/EPOCHS)
        tbCallback = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=BS, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        self.model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.SGD(lr=INIT_LR))
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.6,
                              patience=2, min_lr=0.0000001)
        self.model.fit(self.X_data, self.y_data, epochs=EPOCHS, batch_size=BS, shuffle=True, callbacks=[reduce_lr, tbCallback])
        self.model.save(fileName + '.h5')

    def OutputShape(self):
        print(self.model.output_shape)