# Brando 
# 3/30/2021
# Final Project

from    tensorflow                              import keras
from    atpbar                                  import atpbar
from    Library                                 import FunctionLibrary  as fLib
from    Library                                 import FileSystem       as fs 
from    Library                                 import Logger, InitError, Base, YES, NO
from    tensorflow.keras.preprocessing.image    import ImageDataGenerator
from    sklearn.model_selection                 import train_test_split
from    tensorflow.keras.utils                  import to_categorical
import  pandas              as pd 
import  numpy               as np
import  matplotlib.pyplot   as plt 
import  tkinter             as tk
import  sys
import  os
import  math
import  threading
import  cv2
import  time 

# log = Logger(scriptName=__file__)
gLetter = str()

class TextWindow(tk.Tk):
    """
    https://stackoverflow.com/questions/45397806/update-text-on-a-tkinter-window
    """
    def __init__(self):
        tk.Tk.__init__(self)
        self.label = tk.Label(self, text='Enter text')
        self.label.pack(side = 'top', pady = 5)
        self.button = tk.Button(self, text='stop', command=self.on_button)
        self.button.pack()
        self._number = 0

    def on_button(self):
        self.destroy()

class xASLHandler():
    """
    https://www.kaggle.com/sandeepbhogaraju/sign-language-identifier-100-accuracy-tf-model
    """

    # Data
    _rawTestFile = "data/test/sign.csv"
    _rawTrainFile = "data/train/sign.csv"

    # Cache files
    _testCache = "testImages.cache"
    _trainCache = "trainImages.cache"

    # Label Mapping 
    _labelDictionary = {
        0 : "A",
        1 : "B", 
        2 : "C", 
        3 : "D", 
        4 : "E", 
        5 : "F", 
        6 : "G", 
        7 : "H", 
        8 : "I", 
        9 : "J", 
        10 : "K", 
        11 : "L", 
        12 : "M", 
        13 : "N", 
        14 : "O", 
        15 : "P",
        16 : "Q", 
        17 : "R", 
        18 : "S", 
        19 : "T", 
        20 : "U", 
        21 : "V", 
        22 : "W", 
        23 : "X", 
        24 : "Y", 
        25 : "Z"
    }

    # Specifically for this dataset for ASL 
    _defaultTargetColumn = "label"

    _defaultPaddingForModel = "same"

    _testArrayName = "test"
    _trainArrayName = "train"

    def __init__(self,epochs=10) -> None:
        okayToContinue      = True 
        fullTestFilename    = None
        fullTrainFilename   = None
        temp                = None
        inputShapeModel     = None 
        outputShapeModel    = None
        doTrain             = YES

        self._trainData         = None
        self._testData          = None
        self._testImages        = None
        self._trainImages       = None
        self._imageTestArray    = None
        self._imageTrainArray   = None
        self._model             = None 
        self._reshapeValue      = None
        self._epochs            = epochs
        self._trainGenerator    = None
        self._testGenerator     = None
        self._xTrain            = None
        self._yTrain            = None 
        self._xTest             = None 
        self._yTest             = None 
        self._log               = Logger(scriptName=__file__)
        self._textWindow        = None

        if okayToContinue:
            fullTestFilename = fs.GetFilePath(self._rawTestFile)
            if os.path.exists(fullTestFilename) is False:
                okayToContinue = False
                print("File", fullTestFilename, "does not exist")
            fullTrainFilename = fs.GetFilePath(self._rawTrainFile)
            if os.path.exists(fullTrainFilename) is False:
                okayToContinue = False
                print("File", fullTrainFilename, "does not exist")

        if okayToContinue:
            self._testData = Base(pd.read_csv(fullTestFilename))
            self._trainData = Base(pd.read_csv(fullTrainFilename))
            okayToContinue  = True if self._testData._dataSet.empty is False and self._trainData._dataSet.empty is False else False 

        if okayToContinue:
            okayToContinue = self._testData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        if okayToContinue:
            okayToContinue = self._trainData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        # Organize into reshaped datasets
        if okayToContinue:
            temp = math.sqrt(self._trainData._dataSet[self._trainData._trainColumns].shape[1])
            if math.remainder(temp,1) == 0.0:
                temp                = int(temp)
                self._reshapeValue  = (temp, temp)
            else:
                print("Error in reading data")
                okayToContinue = False
            
        # Get the image arrays for test and train data 
        if okayToContinue:
            self._imageTestArray    = fLib.Load(self._testCache)
            self._imageTrainArray   = fLib.Load(self._trainCache)

            # Test data
            if self._imageTestArray is None:
                print("Image Test cache was not found, creating data...")
                loadImageTestArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._testArrayName,self._testData._dataSet))
                loadImageTestArrayThread.start()

            # Train data 
            if self._imageTrainArray is None:
                print("Image Train cache was not found, creating data...")
                loadImageTrainArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._trainArrayName,self._trainData._dataSet))
                loadImageTrainArrayThread.start()

            if self._imageTrainArray is None:
                loadImageTrainArrayThread.join()

            if self._imageTestArray is None:
                loadImageTestArrayThread.join()

            okayToContinue = True if self._imageTestArray is not None and self._imageTrainArray is not None else False 

        # Prepare variables for the Sequential model 
        if okayToContinue: 
            temp = math.sqrt(len(self._trainData._trainColumns))
            if math.remainder(temp,1) == 0.0:
                temp                = int(temp)
                self._reshapeValue  = (temp, temp, 1)
            else:
                self._log.Error("Incompatable size for input images.  Total pixels for dataset:", len(self._trainData._trainColumns))
                okayToContinue = False

        # Initialize the model 
        if okayToContinue:
            # self.GetIOs()
            self.GetIOs2()
        
            # okayToContinue = self.CreateModel()
            okayToContinue = self.CreateModel2()

        # Train the model 
        if okayToContinue and doTrain:
            # self.Train()
            self.Train2()
            # self.Train3()

        if okayToContinue is False:
            raise InitError(type(self))

    def LoadImageArrayOnThread(self,forArray,dataSet):
        images = []
        for i in atpbar(range(dataSet.shape[0]), name="{} Data".format(forArray)):
            row     = dataSet.iloc[i]
            image   = np.array_split(row[1:],28)
            images.append(image)

        if forArray == "train":
            self._imageTrainArray = np.array(images)
            self._imageTrainArray = np.expand_dims(self._imageTrainArray,axis=3)
            fLib.Save(self._imageTrainArray, self._trainCache)
        elif forArray == "test":
            self._imageTestArray = np.array(images)
            self._imageTestArray = np.expand_dims(self._imageTestArray,axis=3)
            fLib.Save(self._imageTestArray, self._testCache)

    def GetIOs2(self):
        """
        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        self._xTrain    = self._imageTrainArray.astype(float)
        self._yTrain    = self._trainData._dataSet[self._trainData._targetColumns]
        self._xTest     = self._imageTestArray.astype(float)
        self._yTest     = self._testData._dataSet[self._testData._targetColumns].astype(float)

        self._yTrain = to_categorical(self._yTrain, num_classes=25)
        self._yTest = to_categorical(self._yTest, num_classes=25)

        self._xTrain = self._xTrain/255.0
        self._xTest = self._xTest/255.0
        print("x_train shape: ",self._xTrain.shape)
        print("x_test shape: ",self._xTest.shape)    

    def GetIOs(self):
        self._xTrain    = self._imageTrainArray.astype(float)
        self._yTrain    = self._trainData._dataSet[self._trainData._targetColumns].astype(float)
        self._xTest     = self._imageTestArray.astype(float)
        self._yTest     = self._testData._dataSet[self._testData._targetColumns].astype(float)

        self._xTrain, self._X_validate, self._yTrain, self._Y_validate = train_test_split(
            self._xTrain, self._yTrain, test_size = 0.2, random_state = 12345
            )
    
    def CreateModel2(self):
        success = True 
        try:
            self._model = keras.Sequential()
            self._model.add(keras.layers.Conv2D(filters = 8, kernel_size = (5,5),padding = 'Same', 
                            activation ='relu', input_shape = (28,28,1)))
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
            self._model.add(keras.layers.Dropout(0.25))

            self._model.add(keras.layers.Conv2D(filters = 16, kernel_size = (3,3),padding = 'Same', 
                            activation ='relu'))
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2), strides=(2,2)))
            self._model.add(keras.layers.Dropout(0.25))

            self._model.add(keras.layers.Flatten())
            self._model.add(keras.layers.Dense(512, activation = "relu"))
            self._model.add(keras.layers.Dropout(0.5))
            self._model.add(keras.layers.Dense(25, activation = "softmax"))
            optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
            self._model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])
        except Exception as e:
            self._log.Except(e)
            success = False 
        return success

    def Train2(self):
        """
        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        epochs = 3  # for better result increase the epochs
        batch_size = 128
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # dimesion reduction
            rotation_range=10,  # randomly rotate images in the range 5 degrees
            zoom_range = 0.1, # Randomly zoom image 10%
            width_shift_range=0.1,  # randomly shift images horizontally 10%
            height_shift_range=0.1,  # randomly shift images vertically 10%
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        datagen.fit(self._xTrain)
        history = self._model.fit_generator(datagen.flow(self._xTrain, self._yTrain, batch_size=batch_size),
                              epochs = epochs, validation_data = (self._xTest, self._yTest),
                             steps_per_epoch=self._xTrain.shape[0] // batch_size)

    def CreateModel(self):
        success = YES

        # Initialize the model 
        if success:
            self._model = keras.Sequential()
            try:
                outputShapeModel = len(self._labelDictionary)

                self._model.add(keras.layers.Conv2D(16, (3,3), padding='same', activation=keras.activations.relu,input_shape=self._reshapeValue))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Conv2D(32, (3,3), padding='same', activation=keras.activations.relu))
                self._model.add(keras.layers.MaxPooling2D((2,2)))
                # self._model.add(keras.layers.Conv2D(64, (3,3), padding='same', activation=keras.activations.relu))
                # self._model.add(keras.layers.MaxPooling2D((2,2)))
                # self._model.add(keras.layers.Conv2D(128, (3,3), padding='same', activation=keras.activations.relu))
                # self._model.add(keras.layers.MaxPooling2D((2,2)))
                self._model.add(keras.layers.Flatten())
                self._model.add(keras.layers.Dense(64, activation=keras.activations.relu))
                self._model.add(keras.layers.Dense(outputShapeModel, activation=keras.activations.softmax))
            except TypeError as e:
                self._log.Fatal("Could not build keras model")
                self._log.Except(e)
                success = False
            except ValueError as e:
                self._log.Fatal("Could not build keras model")
                self._log.Except(e)
                success = False
            except Exception as e:
                self._log.Fatal("Unknown exception")
                self._log.Except(e)
                success = False
        
        # Compile the model 
        if success:
            try:
                self._model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics = ['accuracy'])
                self._model.summary()
            except ValueError as e:
                self._log.Fatal("Could not compile keras model")
                self._log.Except(e)
                success = False
            except Exception as e:
                self._log.Fatal("Unknown exception")
                self._log.Except(e)
                success = False
        
        return success

    def Train(self):
        train_datagen = ImageDataGenerator(
            rescale=1/255,rotation_range=45, width_shift_range=0.25,
            height_shift_range=0.15,shear_range=0.15, zoom_range=0.2, 
            fill_mode='nearest'
        )
        test_datagen    = ImageDataGenerator(rescale=1/255)
        valid_datagen   = ImageDataGenerator(rescale=1/255)

        self._trainGenerator    = train_datagen.flow(self._xTrain, self._yTrain, batch_size=32)
        self._testGenerator     = test_datagen.flow(self._xTest,self._yTest,batch_size=32)
        valid_generator         = valid_datagen.flow(self._X_validate,self._Y_validate,batch_size=32)
        
        self._model.fit(
            self._trainGenerator,
            epochs=self._epochs,
            validation_data=valid_generator,
            callbacks = [
                keras.callbacks.EarlyStopping(monitor='loss', patience=10),
                keras.callbacks.ModelCheckpoint(
                    filepath='/kaggle/working/',
                    monitor='val_accuracy',
                    save_best_only=True
                )
            ]
        )

        pred = self._model.predict(self._testGenerator)

    def Train3(self):    
        print(self._xTrain.shape)
        print(self._yTrain.shape)
        self._model.fit(self._xTrain, self._yTrain, epochs=self._epochs)

    def Run(self):
        """
        To Plot
        ------
        plt.imshow(self._imageTestArray[i])
        """
        # define a video capture object
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        runWindowThread = threading.Thread(target=self.RunWindow)
        runWindowThread.daemon = True 
        runWindowThread.start()
        
        try:
            while(True):
                
                # Capture the video frame
                # by frame
                ret, frame = vid.read()

                # I can start a thread here that processes the frames
            
                # Display the resulting frame
                pred = self.GetPrediction(frame)
                self.UpdateText(pred) # TODO update with predictions 
                # self.UpdateText(time.strftime("%S")) # TODO update with predictions 
                cv2.imshow('title', frame)
                
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            pass 
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()
        runWindowThread.join()

    def GetPrediction(self,frame: np) -> str:
        result = str() 

        # Resize 
        res = cv2.resize(frame,(28,28),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        res = np.expand_dims(res[:,:,0], axis=-1)
        res = np.expand_dims(res, axis=0)
        res = res.astype(float)

        inputData = ImageDataGenerator(rescale=1/255)
        input     = inputData.flow(res)
        array = self._model.predict(input)
        index = np.argmax(array)
        result = self._labelDictionary[index]
        return result

    def RunWindow(self):
        self._textWindow = TextWindow()
        self._textWindow.resizable(width=True, height=True)
        self._textWindow.geometry('{}x{}'.format(100, 90))
        self._textWindow.mainloop()

    def UpdateText(self, value: str):
        if self._textWindow is not None: 
            self._textWindow.label['text'] = value

def main():
    """
    Final Project
    =============
    Author: Brando 

    Oultine 
    -------------
    The final image recognition project is following the structure of Chapter 2, that is the main steps to consider are:

    1. Look at the big picture.
    2. Get the data.
    3. Discover and visualize the data to gain insights.
    4. Prepare the data for Machine Learning algorithms.
    5. Select a model and train it.
    6. Fine-tune your model.

    """
    signLangHandler = xASLHandler(1)
    signLangHandler.Run()

if __name__ == "__main__":
    main()
    
