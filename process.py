import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
import numpy as np
import os
from win10toast import ToastNotifier

class ThrombosisPredictor:

    def __init__(self) -> None:
        answer = input('Would you like to train a new model? Y/n \n')
        if answer == 'Y' or answer == 'y':
            X_train, X_test, Y_train, Y_test = self.processRawData()
            maxIteration = int(input('Please enter maximum iteration number: \n'))
            trainedModel = self.fitModel(X_train, X_test, Y_train, Y_test, maxIteration)
            self.predictExample(trainedModel)
            saveOrNo = input('Would you like to save the model? Y/n \n')
            if saveOrNo == 'Y' or saveOrNo == 'y':
                modelName = input('Enter a name for your model: \n')
                self.saveModel(model=trainedModel, modelName=modelName)
        else:
            fileName = input('Please enter the file name of a saved model: \n')
            loadedModel = self.loadModel(fileName)
            self.predictExample(loadedModel)

    def saveModel(self, model: MLPClassifier, modelName):
        currentDirectory = os.getcwd()
        folderPath = os.path.join(currentDirectory, 'Models')
        if not os.path.exists(folderPath):
            os.mkdir(folderPath)
        joblib.dump(model, filename=os.path.join(folderPath, modelName+'.sav'))
        
    def loadModel(self, modelName) -> MLPClassifier:
        currentDirectory = os.getcwd()
        folderPath = os.path.join(currentDirectory, 'Models')
        return joblib.load(filename=os.path.join(folderPath, modelName))

    def processRawData(self):
        excelfile = pd.read_csv('files/diag_surg_complic.csv')
        excel_data = excelfile[[
            'time_diag_surg',
            'bleeding',
            'heartburn',
            'fever',
            'infection',
            'pain',
            'nausea',
            'confusion',
            'high_blood_pressure',
            'shock',
            'thrombosis',
            'number_of_complications']]

        result_data = excel_data.iloc[:,0:10]
        health_data = excel_data.iloc[:,10]

        X_train, X_test, Y_train, Y_test = train_test_split(
            health_data,
            result_data,
            test_size=0.2
            )
        return X_train, X_test, Y_train, Y_test
          
    def fitModel(self, X_train, X_test, Y_train, Y_test, maxIteration: int) -> MLPClassifier:
        mlp = MLPClassifier(hidden_layer_sizes=(250, 200, 150), max_iter=maxIteration, alpha=1e-4,
                    solver='lbfgs', verbose=10, activation='logistic', random_state=1,
                    learning_rate_init=.1, learning_rate='adaptive')

        temp = X_train
        X_train = Y_train
        Y_train = temp

        temp = X_test
        X_test = Y_test
        Y_test = temp

        mlp.fit(X_train, Y_train)

        y_prediction = mlp.predict(X_test)

        print("Training set score: %f" % mlp.score(X_train, Y_train))
        print("Test set score: %f" % mlp.score(X_train, Y_train))

        accuracy = accuracy_score(Y_test, y_prediction)
        print('Accuracy: %f' % accuracy)

        conf = confusion_matrix(Y_test, y_prediction)

        print(conf)
        
        toast = ToastNotifier()
        toast.show_toast('Model training complete!',
                         'Model successfully trained with %i iterations.' %maxIteration,
                         duration=10,
                         icon_path='success.ico')
        
        return mlp
  
    def predictExample(self, mlp: MLPClassifier):
        print('----------------------------------------------')

        print('Example prediction:')
        arrayLike = [
            [126,0,0,1,1,1,0,0,1,0],
            [40,0,1,0,0,1,0,0,1,0],
            [3,0,0,1,0,0,0,0,1,0],
            [32,0,0,0,1,0,0,0,0,0],
            [162,0,0,1,0,1,0,0,0,0],
            [25,0,0,0,0,1,1,0,1,0],
            [56,0,0,0,0,1,0,0,0,0],
            [15,1,0,0,1,0,1,0,1,0],
            [6,0,1,0,0,1,0,0,0,0]]

        df = pd.DataFrame(arrayLike, columns=['time_diag_surg',
                                    'bleeding',
                                    'heartburn',
                                    'fever',
                                    'infection',
                                    'pain',
                                    'nausea',
                                    'confusion',
                                    'high_blood_pressure',
                                    'shock'])
        proba = mlp.predict(df)
        accuracy = accuracy_score([1,0,0,1,0,1,1,1,1], proba)

        print('Predicted data: ', proba)
        print('Real data: [1 0 0 1 0 1 1 1 1]')
        print('Accuracy: %f' % accuracy)

    
if __name__ == '__main__':
    ThrombosisPredictor()



