from res.imports.Imports import*
from Utilities.data_splitting import DataSplitting


def RemoveDup(df1):
    # Checking for Any Duplicate values in the dataset.
    print("Count of Duplicate Values : ", df1.duplicated().sum())

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Removing Duplicate rows from Dataframe.
    df2 = df1.drop_duplicates(keep=False)
    print("Removing Duplicate Rows ... ")

    # After removing duplicate rows.
    print("Count of Duplicate Values : ", df2.duplicated().sum())
    print("Count of Null values : ", df2.isnull().sum().sum(), "\n")
    print("Count of Null values : \n", df2.isnull().sum(), "\n")
    print("Number of unique Rows : ", len(df2), "\n")
    print("Data Cleaning Completed...")

    print(
        "<----------------------------------------------------------------------------------------------------------->")

    # Calling DataSplitting Function

    li_Fs_Data = DataSplitting(df2)
    return li_Fs_Data
