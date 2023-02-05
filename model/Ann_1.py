from res.imports.Imports import *
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def ArtificialNN_1(x_train, y_train, x_test, y_test,sc):

    # initializing ANN
    ann = tf.keras.models.Sequential()

    # Add the input layer and first hidden layer
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=12, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

    ann.add(tf.keras.layers.Dense(units=1, activation='relu'))

    ann.compile(optimizer='adam', loss='mean_squared_error')

    # Converting the data of fit
    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    y_train.resize((len(x_train), 1))
    y_test.resize((len(x_test), 1))
    y_test[y_test == 0] = np.nan

    mask = np.isnan(y_test)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    y_test[mask] = y_test[np.nonzero(mask)[0], idx[mask]]
    y_test[890:904] = 585000

    ann.fit(x_train, y_train, batch_size=32, epochs=100)

    y_pred1= ann.predict(sc.transform([[4.0, 2.5, 1850.0, 5040.0, 1.0, 0.0, 0.0, 3, 1230.0, 620.0, 29, 31, 59]]))
    y_pred2 = ann.predict(sc.transform([[4.0, 3.25, 3990.0 ,9786.0, 2.0, 0.0, 0.0, 3, 3990.0 ,0.0, 102, 27, 31]]))

    return [y_pred1, y_pred2]

