import numpy as np
import pandas as pd

# Importing functions from user defined files.
from Utilities.feteching_data import Fetching_Data
from Utilities.remove_dup import RemoveDup
from Utilities.remove_null import RemoveNull
from Utilities.data_splitting import DataSplitting
from Utilities.remove_dup import RemoveDup
from model.Ann_1 import ArtificialNN_1
from model.Ann_2 import ArtificialNN_2
from model.Ann_3 import ArtificialNN_3

import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn.preprocessing import StandardScaler

# For Splitting Data into Training set and Test set.
from sklearn.model_selection import train_test_split

# For Applying Feature Scaling.
from sklearn.preprocessing import StandardScaler