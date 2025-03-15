from django.shortcuts import render

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


def home(request):
    return render(request, 'home.html')


def predict(request):
    return render(request, 'predict.html')


def result(request):
    df = pd.read_csv(r"C:\Users\lymon\Desktop\Millicent Final\healthcare-dataset-stroke-data.csv")

    df.duplicated().sum()

    df.dropna(axis=0, inplace=True)

    df = df.rename(columns={'ever_married': 'Married'})

    le = LabelEncoder()

    df.avg_glucose_level = df['avg_glucose_level'].astype('int64')
    df.bmi = df['bmi'].astype('int64')

    df["gender"] = le.fit_transform(df["gender"])
    df["Married"] = le.fit_transform(df["Married"])
    df["work_type"] = le.fit_transform(df["work_type"])
    df["Residence_type"] = le.fit_transform(df["Residence_type"])
    df["smoking_status"] = le.fit_transform(df["smoking_status"])

    X = df.drop('stroke', axis=1)
    y = df['stroke']

    from imblearn.over_sampling import SMOTE, SVMSMOTE, RandomOverSampler
    oversamp = RandomOverSampler(random_state=0)
    oversamp = SMOTE(n_jobs=-1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=y)
    X_train, y_train = oversamp.fit_resample(X_train, y_train)

    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model_svm = SVC(kernel='linear', random_state=0)
    model_svm.fit(X_train, y_train)

    y_pred = model_svm.predict(X_test)

    val1 = float(request.GET['n1'])
    val2 = float(request.GET['n2'])
    val3 = float(request.GET['n3'])
    val4 = float(request.GET['n4'])
    val5 = float(request.GET['n5'])
    val6 = float(request.GET['n6'])
    val7 = float(request.GET['n7'])
    val8 = float(request.GET['n8'])
    val9 = float(request.GET['n9'])
    val10 = float(request.GET['n10'])
    val11 = float(request.GET['n11'])

    pred = model_svm.predict([[val1, val2, val3, val4, val5, val6, val7, val8, val9, val10, val11]])

    result1 = ""
    if pred == [1]:
        result1 = "The person have stroke"
    else:
        result1 = "The person have no stroke"

    return render(request, 'predict.html', {"result2": result1})
