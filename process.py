import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


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
    'number_of_complications'
    ]]

result_data = excel_data.iloc[:,0:10]
health_data = excel_data.iloc[:,10]

X_train, X_test, Y_train, Y_test = train_test_split(
    health_data,
    result_data,
    test_size=0.2
    )


mlp = MLPClassifier(hidden_layer_sizes=(250, 200, 150 ), max_iter=10000, alpha=1e-4,
                    solver='lbfgs', verbose=10, activation='relu', random_state=1,
                    learning_rate_init=.1)

temp = X_train
X_train = Y_train
Y_train = temp

temp = X_test
X_test = Y_test
Y_test = temp

print('----------------------------------------------')
print(X_train.shape)
print(Y_train.shape)
print('----------------------------------------------')

mlp.fit(X_train, Y_train)

y_prediction = mlp.predict(X_test)

print("Training set score: %f" % mlp.score(X_train, Y_train))
print("Test set score: %f" % mlp.score(X_train, Y_train))

accuracy = accuracy_score(Y_test, y_prediction)
print('Accuracy: %f' % accuracy)

conf = confusion_matrix(Y_test, y_prediction)

print(conf)

print('----------------------------------------------')

print('Prediction:')
arrayLike = [
    [126,0,0,1,1,1,0,0,1,0],
    [40,0,1,0,0,1,0,0,1,0],
    [3,0,0,1,0,0,0,0,1,0],
    [32,0,0,0,1,0,0,0,0,0],
    [162,0,0,1,0,1,0,0,0,0],
    [25,0,0,0,0,1,1,0,1,0],
    [56,0,0,0,0,1,0,0,0,0],
    [15,1,0,0,1,0,1,0,1,0],
    [6,0,1,0,0,1,0,0,0,0]
    ]

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
print(proba)
print(accuracy_score([1,0,0,1,0,1,1,1,1], proba))