from res.imports.Imports import *

def LabelEncoder(data, num):

    def LE_X_train(X_train):
        from sklearn.preprocessing import LabelEncoder
        LE = LabelEncoder()
        X_train[:, 10] = LE.fit_transform(X_train[:, 10])
        print("Number of Unique Values Encoded for Year_Build : ",len(np.unique(LE.classes_)))
        X_train[:, 11] = LE.fit_transform(X_train[:, 11])
        print("Number of Unique Values Encoded for City : ", len(np.unique(LE.classes_)))
        X_train[:, 12] = LE.fit_transform(X_train[:, 12])
        print("Number of Unique Values Encoded for State Zip : ", len(np.unique(LE.classes_)))

        return X_train

    def LE_X_Test(X_test):
        from sklearn.preprocessing import LabelEncoder
        LE = LabelEncoder()
        X_test[:, 10] = LE.fit_transform(X_test[:, 10])
        X_test[:, 11] = LE.fit_transform(X_test[:, 11])
        X_test[:, 12] = LE.fit_transform(X_test[:, 12])

        return X_test

    if num == 0:
        data = LE_X_train(data)
        return data

    else:
        data = LE_X_Test(data)
        return data






