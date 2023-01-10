import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.options.display.float_format = '{:.1f}'.format
pd.options.display.max_rows = 10

# Load dataset
training_df = pd.read_csv("Iris.csv")

# Build model
model = None


def build_model(learning_rate):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    return model

# Model training, hyperparameters


def train_model(model, df, feature, label, epochs, batch_size):
    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=batch_size,
                        epochs=epochs)

    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    epochs = history.epoch
    hist = pd.DataFrame(history.history)

    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse

# Plotting loss curve


def plot_loss_curve(epochs, rmse):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    plt.plot(epochs, rmse, label="Loss")

    plt.show()


# Setting hyperparameters
learning_rate = number
epochs = number
batch_size = number

# Setting features
feature = "SepalLengthCm"






