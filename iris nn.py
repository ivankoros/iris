import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Reading, dropping ID column
iris_df = pd.read_csv('iris.csv')
iris_df = iris_df.drop(columns=['Id'])

# Making columns into Z scores
iris_df = iris_df.apply(lambda x: (x - x.mean()) / x.std(), axis=0)

# Encoding "species" column and setting as a label
label_encoder = LabelEncoder()
iris_df['Species'] = label_encoder.fit_transform(iris_df['Species'])

# Setting features and creating feature layer
feature_columns = []
for x in iris_df.columns:
    if x != 'Species':
        feature_columns.append(tf.feature_column.numeric_column(x))

feature_layer = tf.keras.layers.DenseFeatures(feature_columns)


def create_model(learning_rate, feature_layer):
    inputs = tf.keras.Input(shape=(4,))
    x = feature_layer(inputs)
    x = tf.keras.layers.Dense(10, activation='relu')(inputs)
    x = tf.keras.layers.Dense(10, activation='relu')(x)
    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

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

# Set label
label_name = "Species"

# Build model
my_model = create_model(learning_rate, feature_layer)

# Train data
epochs, mse = train_model(my_model, iris_df, epochs, label_name, batch_size)
plot_loss_curve(epochs, mse)

