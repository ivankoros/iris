import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns

iris_df = pd.read_csv('iris.csv')
iris_df = iris_df.drop(columns=['Id'])

print(iris_df.head())

feature_columns = []
for x in iris_df.columns:
    feature_columns.append(tf.feature_column.numeric_column(x))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


# Plot loss curve


def plot_loss_curve(epochs, mse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")

    plt.plot(epochs, mse, label="Loss")
    plt.legend()
    plt.ylim([mse.min() * 0.95, mse.max() * 1.05])
    plt.show()


def create_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)

    model.add(tf.keras.layers.Dense(units=20,
                                    activation='relu',
                                    name='Hidden1'))

    model.add(tf.keras.layers.Dense(units=12,
                                    activation='relu',
                                    name='Hidden2'))

    model.add(tf.keras.layers.Dense(units=1,
                                    name='Output'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.MeanSquaredError()])

    return model


def train_model(model, dataset, epochs, label_name, batch_size=None):
    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))

    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    epochs = history.epoch

    hist = pd.DataFrame(history.history)
    mse = hist["mean_squared_error"]

    return epochs, mse

# Setting hyperparameters

learning_rate = 0.01
epochs = 100
batch_size  = 1000

label_name = "Species"

my_model = create_model(learning_rate, feature_layer)

epochs, mse = train_model(my_model, iris_df, epochs, label_name, batch_size)
plot_loss_curve(epochs, mse)

