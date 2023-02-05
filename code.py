from res.imports.Imports import *

class Code:
    def __init__(self):
        super().__init__()

        ### <--- Data PreProcessing Phase. --->
        """
        1. Starting with Fetching data.
        2. Selecting Needed data.
        3. Dealing with Removing Null Values.
        4. Dealing with Duplicate Values.
        5. Splitting Data.   
        """
        # Assigning essentials
        data = Fetching_Data()
        X_train = data[0]
        X_test = data[1]
        Y_train = data[2]
        Y_test = data[3]
        sc = data[4]

        # Printing training and test data after preprocessing
        print("\n X_train : ", "\n", X_train[:5], "\n")
        print("X_test : ", "\n", X_test[:5], "\n")
        print("Y_train :", "\n", Y_train[0:5], "\n")
        print("Y_test : ", "\n", Y_test[:, :5], "\n")

        ### <--- Model Selection Phase. --->

        relu = ArtificialNN_1(X_train, Y_train, X_test, Y_test, sc)

        elu = ArtificialNN_2(X_train, Y_train, X_test, Y_test, sc)

        mix = ArtificialNN_3(X_train, Y_train, X_test, Y_test, sc)

        # printing the predicted values.
        print("\n \n Real Price Value : 480000.00 , Predicted Value 1 : ", relu[0],"Predicted value 2 : ",elu[0],
              "Predicted value 3 : ", mix[0])
        print("\n Real Price Value : 860000.00 , Predicted Value 1 : ", relu[1], "Predicted value 2 : ",elu[1],
              "Predicted value 3 : ", mix[1], "\n")













