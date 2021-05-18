"""
main.py
================
Contains xASLHandler class

Todo
-------------------
- Properly exit out of application 
"""
# Brando 
# 3/30/2021
# Final Project

from    tensorflow                              import keras
from    atpbar                                  import atpbar
from tensorflow.python.keras.callbacks import Callback
from    Library                                 import FunctionLibrary  as fLib
from    Library                                 import FileSystem       as fs 
from    Library                                 import Logger, InitError, Base, YES, NO, GetPercentage
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

class PredictionOutput(tk.Tk):
    """
    PredictionOutput
    ==================
    Shows the ASLHandler's prediction and confidence values 
    https://stackoverflow.com/questions/45397806/update-text-on-a-tkinter-window
    """
    def __init__(self, stopRecordingCallback: Callback):
        tk.Tk.__init__(self)

        # Create prediction label 
        self._predictionLabel = tk.Label(self, text='Prediction')
        self._predictionLabel.pack(side = 'top', pady = 5)

        # Create Confidence label 
        self._confidenceLabel = tk.Label(self, text='Confidence')
        self._confidenceLabel.pack(side = 'top', pady = 5)

        # Stop button 
        self.button = tk.Button(self, text='Stop', command=self.on_button)
        self.button.pack()

        # Save callback function
        self._stopRecordingCallback = stopRecordingCallback

    def on_button(self):
        self.destroy()
        self._stopRecordingCallback() # Kill the recording

class xASLHandler():
    """
    xASLHandler
    =============
    """

    # Data
    _rawTestFile    = "data/test/sign.csv"
    _rawTrainFile   = "data/train/sign.csv"

    # Cache files
    _cacheDir           = "cache"
    _testCache          = _cacheDir + "/testImages.cache"
    _trainCache         = _cacheDir + "/trainImages.cache"
    _targetTestCache    = _cacheDir + "/targetTest.cache"
    _targetTrainCache   = _cacheDir + "/targetTrain.cache"

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

    _testArrayName  = "test"
    _trainArrayName = "train"

    def __init__(self,epochs=10) -> None:
        okayToContinue      = True 
        fullTestFilename    = None
        fullTrainFilename   = None
        temp                = None
        doTrain             = YES
        tempList            = list()

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
        self._keepRecording     = True

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
            self._testCache         = fs.GetFilePath(self._testCache)
            self._trainCache        = fs.GetFilePath(self._trainCache)
            self._targetTestCache   = fs.GetFilePath(self._targetTestCache)
            self._targetTrainCache  = fs.GetFilePath(self._targetTrainCache)
            tempList = [self._testCache, self._trainCache, self._targetTestCache, self._targetTrainCache]
            for cacheFile in tempList:
                okayToContinue = fs.CreatePath(cacheFile,True)
                if okayToContinue is False:
                    self._log.Fatal("Could not get path to cache:", cacheFile)
                    break

        if okayToContinue:
            self._testData  = Base(pd.read_csv(fullTestFilename))
            self._trainData = Base(pd.read_csv(fullTrainFilename))
            okayToContinue  = True if self._testData._dataSet.empty is False and self._trainData._dataSet.empty is False else False 

        if okayToContinue:
            okayToContinue = self._testData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        if okayToContinue:
            okayToContinue = self._trainData.CreateTrainAndTargetColumns(targetColumns=[self._defaultTargetColumn])

        # Create the output window 
        if okayToContinue:
            self._textWindow = PredictionOutput(stopRecordingCallback=self.StopRecording)
            self._textWindow.resizable(width=True, height=True)
            self._textWindow.geometry('{}x{}'.format(200, 200))
            if self._textWindow is None:
                self._log.Fatal("Could not create output window")
                okayToContinue = False 

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
            self._yTest             = fLib.Load(self._targetTestCache)
            self._yTrain            = fLib.Load(self._targetTrainCache)

            # Test data
            if self._imageTestArray is None or self._yTest is None:
                print("Image Test cache was not found, creating data...")
                loadImageTestArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._testArrayName,self._testData._dataSet))
                loadImageTestArrayThread.start()

            # Train data 
            if self._imageTrainArray is None or self._yTrain is None:
                print("Image Train cache was not found, creating data...")
                loadImageTrainArrayThread = threading.Thread(target=self.LoadImageArrayOnThread, args=(self._trainArrayName,self._trainData._dataSet))
                loadImageTrainArrayThread.start()

            if self._imageTrainArray is None or self._yTrain is None:
                loadImageTrainArrayThread.join()

            if self._imageTestArray is None or self._yTest is None:
                loadImageTestArrayThread.join()

            if self._imageTestArray is None and self._imageTrainArray is None:
                okayToContinue = False 
            elif self._yTest is None and self._yTrain is None:
                okayToContinue = False
            elif len(self._yTrain) != len(self._imageTrainArray):
                okayToContinue = False
            elif len(self._yTest) != len(self._imageTestArray):
                okayToContinue = False

            if okayToContinue is False:
                self._log.Error("Could not get train or test data")

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
            self.GetIOs2()
            okayToContinue = self.CreateModel2()

        # Train the model 
        if okayToContinue and doTrain:
            self.Train2()

        if okayToContinue is False:
            raise InitError(type(self))

    def LoadImageArrayOnThread(self,forArray,dataSet):
        """
        LoadImageArrayOnThread
        ===========
        Generates the image data

        Could I flip the images?  
        """
        success     = True 
        images      = []
        imageArray  = None 
        cacheFile   = str()
        reversed    = None
        targetArray = list()
        targetCache = str()

        if success:
            for i in atpbar(range(dataSet.shape[0]), name="{} Data".format(forArray)):
                row = dataSet.iloc[i]

                # Normal
                image = np.array_split(row[1:],28)
                images.append(image)
                targetArray.append(row[self._trainData._targetColumns]) # Trusting that test and train data both have same target name

                # # Mirrored image
                # reversed = row[::-1]
                # image = np.array_split(reversed[:-1],28)
                # images.append(image)
                # targetArray.append(row[self._trainData._targetColumns]) # Trusting that test and train data both have same target name

            if len(images) == 0:
                success = False
                self._log.Fatal("Could not create image array for {} data".format(forArray))

        if success:
            imageArray = np.array(images)
            imageArray = np.expand_dims(imageArray,axis=3)
            if imageArray is None:
                success = False 
                self._log.Fatal("Unexpected outcome from numpy array generation")

        if success: 
            if forArray == self._trainArrayName:
                cacheFile               = self._trainCache
                self._imageTrainArray   = imageArray

                targetCache     = self._targetTrainCache
                self._yTrain    = targetArray
            elif forArray == self._testArrayName:
                cacheFile               = self._testCache
                self._imageTestArray    = imageArray

                targetCache = self._targetTestCache
                self._yTest = targetArray

            fLib.Save(imageArray, cacheFile)
            fLib.Save(targetArray, targetCache)

    def GetIOs2(self):
        """
        GetIOs2
        ============
        Second revision of the test and train set  

        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        self._xTrain    = self._imageTrainArray.astype(float)
        self._xTest     = self._imageTestArray.astype(float)

        self._yTrain    = to_categorical(self._yTrain, num_classes=25)
        self._yTest     = to_categorical(self._yTest, num_classes=25)

        self._xTrain    = self._xTrain/255.0
        self._xTest     = self._xTest/255.0

    def CreateModel2(self):
        """ 
        CreateModel2
        ============
        Second revision of the model 

        https://www.kaggle.com/a7madmostafa/sign-mnist-with-cnn-100-accuracy 
        """
        success = True 
        try:
            # Establish Sequential model 
            self._model = keras.Sequential()

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters     = 128, 
                    kernel_size = (3,3),
                    strides     = 1,
                    padding     = 'Same', 
                    activation  = 'relu', 
                    input_shape = self._reshapeValue
            ))

            # MaxPool2D
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters     = 64, 
                    kernel_size = (3,3),
                    strides     = 1,
                    padding     = 'Same', 
                    activation  ='relu'
            ))

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters     = 64, 
                    kernel_size = (3,3),
                    strides     = 1,
                    padding     = 'Same', 
                    activation  ='relu'
            ))

            # MaxPool2D
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

            # Conv2D
            self._model.add(
                keras.layers.Conv2D(
                    filters     = 25, 
                    kernel_size = (3,3),
                    strides     = 1,
                    padding     = 'Same', 
                    activation  ='relu'
            ))

            # MaxPool2D
            self._model.add(keras.layers.MaxPool2D(pool_size=(2,2)))

            self._model.add(keras.layers.Flatten())

            self._model.add(keras.layers.Dense(512, activation = "relu"))

            self._model.add(keras.layers.Dropout(0.2))

            # 25 outputs
            self._model.add(keras.layers.Dense(25, activation = "softmax"))

            self._model.compile(
                optimizer   = "adam" , 
                loss        = "categorical_crossentropy", 
                metrics     = ["accuracy"]
            )

            # Print Summary
            self._model.summary()

        except Exception as e:
            self._log.Except(e)
            success = False 
        return success

    def Train2(self):
        """
        Train2
        ===========
        2nd revision of the training 

        References
        -------------
        https://www.kaggle.com/hkubra/mnist-cnn-with-keras-99-accuracy
        """
        batch_size = 128
        datagen = ImageDataGenerator(
            featurewise_center              = False, 
            samplewise_center               = False, 
            featurewise_std_normalization   = False, 
            samplewise_std_normalization    = False, 
            zca_whitening                   = False, 
            rotation_range                  = 10, 
            zoom_range                      = 0.1, 
            width_shift_range               = 0.1, 
            height_shift_range              = 0.1,
            horizontal_flip                 = False, 
            vertical_flip                   = False
        ) 

        datagen.fit(self._xTrain)
        _ = self._model.fit_generator(
            datagen.flow(
                self._xTrain, 
                self._yTrain, 
                batch_size = batch_size
            ),
            epochs          = self._epochs, 
            validation_data = (self._xTest, self._yTest),
            steps_per_epoch = self._xTrain.shape[0] // batch_size
        )

    def Record(self):
        """
        Record 
        ================
        Uses cv2 to record user's hands
        """
        self._keepRecording = True 

        # define a video capture object
        vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        try:
            while self._keepRecording:
                
                # Capture the video frame
                # by frame
                _, frame = vid.read()

                # Display the resulting frame
                pred, conf = self.GetPrediction(frame)
                self.UpdateText(pred,conf)
                cv2.imshow('title', frame)
                
                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except KeyboardInterrupt:
            self._log.Warn("User interrupted")
            pass 
        
        vid.release() # After the loop release the cap object
        cv2.destroyAllWindows() # Destroy all the windows

    def GetPrediction(self,frame: np) -> str:
        """
        GetPrediction
        ================
        Runs the model's prediction on the frame generated by the recording window 
        """
        result = str() 
        conf = float()

        # Resize 
        res = cv2.resize(frame,(28,28),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        res = np.expand_dims(res[:,:,0], axis=-1)
        res = np.expand_dims(res, axis=0)
        res = res.astype(float)

        # Normalize 
        inputData   = ImageDataGenerator(rescale=1/255)
        input       = inputData.flow(res)

        # Predict
        array   = self._model.predict(input)
        index   = np.argmax(array)
        conf    = array[0][index]
        result  = self._labelDictionary[index]

        return result, conf

    def Run(self):
        """
        Run 
        =================
        Main loop 
            - Runs record and output windows 
        """
        # Start the recording thread 
        recordThread         = threading.Thread(target=self.Record)
        recordThread.daemon  = True 
        recordThread.start()

        # Start the output window 
        self._textWindow.mainloop()

    def UpdateText(self, value: str, confidence: float):
        """
        UpdateText
        ===============
        Updates the labels in the output window 
        """
        if self._textWindow is not None: 
            self._textWindow._predictionLabel['text'] = "Prediction: {}".format(value)
            self._textWindow._confidenceLabel['text'] = "Confidence: {}%".format(GetPercentage(confidence))

    def StopRecording(self):
        """
        StopRecording
        =================
        Negates the record flag to signal the cv2 window to stop recording 
        """
        self._keepRecording = False 

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
    signLangHandler = xASLHandler(epochs=5)
    signLangHandler.Run()

if __name__ == "__main__":
    main()
    
