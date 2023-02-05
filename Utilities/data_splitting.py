from Utilities.feature_scaling import apply_feature_scaling
from res.imports.Imports import *
from Utilities.encoding_data import LabelEncoder

def DataSplitting(df2):
        # Splitting Data into Dependent and Independent variables.
        X = df2.iloc[:, 2:-1].values  # Independent
        pd.set_option('display.max_columns', None)
        Y = df2.iloc[:, 1].values  # Dependent

        print("Deleting Columns which are irrevalent for your Predictive Algorithm. \n")
        X = np.delete(X, [12, 11], axis=1)
        print("Columns deleted :  yr_renovated , Stree Name\n")



        # Splitting Data into Training set and Test set.
        from sklearn.model_selection import train_test_split
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        print("Size of X_train: ", len(X_train), "\n", X_train[:5], "\n", ", Size of X_test: ", len(X_test), "\n", X_test[:5], "\n",)
        print("Size of Y_train: ", len(Y_train), "\n", Y_train[:5], "\n", ", Size of Y_test: ", len(Y_test), "\n", Y_test[:5], "\n",)

        # Replaceing 0 with NAN so we can impute the values easily.
        Y_test[Y_test == 0] = np.nan

        # Impute Function to replace all np.nan values with mean value of array to get some Consistent data.
        def ImputeNan(y):
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
                imputer.fit([y])
                y = imputer.transform([y])

                return y

        # Calling Imputer function on Dependent data.
        Y_test = ImputeNan(Y_test)
        Y_train = ImputeNan(Y_train)


        print("<----------------------------------------------------------------------------------------------------------->")


        print("Encoding Categorical Data into Numerical Data ...")
        X_train = LabelEncoder(X_train, 0)
        X_test = LabelEncoder(X_test, 1)

        li_Fs_Data = apply_feature_scaling(X_train, X_test, Y_train, Y_test)

        return li_Fs_Data

