from res.imports.Imports import *
from Utilities.encoding_data import LabelEncoder

def apply_feature_scaling(X_train, X_test, Y_train, Y_test):

        # Applying Feature Scaling.
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        li_Fs_Data = [X_train, X_test, Y_train, Y_test, sc]

        return li_Fs_Data