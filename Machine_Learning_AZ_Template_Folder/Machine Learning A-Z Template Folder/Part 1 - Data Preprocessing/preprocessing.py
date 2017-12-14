import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv("Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
imputer = imputer.fit(x[:, 1:3])
x[:, 1:3] = imputer.transform(x[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
x[:, 0] = labelEncoder_X.fit_transform(x[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
x = onehotencoder.fit_transform(x).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
SC_x = StandardScaler()
x_train = SC_x.fit_transform(x_train)
x_test = SC_x.transform(x_test)


