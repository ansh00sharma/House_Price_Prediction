from res.imports.Imports import*
from Utilities.remove_null import RemoveNull


def Fetching_Data():
    # Reading data from dataset.

    df = pd.read_csv('res/dataset/data.csv')

    # Displaying Columns name to Work on, Alongside with Dataset Dimensionality.
    print("column Names of Dataset :")
    print(df.columns)
    col_names = list(df)
    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Converting data into Dataframe.
    print("rows X columns : ", len(df), "X", len(col_names))

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Calling Remove_Null function.
    li_Fs_Data = RemoveNull(df)
    return li_Fs_Data
