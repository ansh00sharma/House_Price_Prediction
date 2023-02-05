from res.imports.Imports import*
from Utilities.remove_dup import RemoveDup


def RemoveNull(df):
    # Checking for Null values in the dataset.
    print("Null count Table for the dataset ", df.isnull().sum().sum(), "\n")
    print(df.isnull().sum())

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Dropping Rows with Null values.
    df1 = df.dropna(axis=0, how="any", thresh=None, subset=None)
    print("Rows dropped : ", df1.isnull().sum().sum())
    print(df1)

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Calling RemoveDup Function.
    li_Fs_Data = RemoveDup(df1)
    return li_Fs_Data
