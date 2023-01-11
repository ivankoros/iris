import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import seaborn as sns

iris_df = load_iris()
iris_df = pd.DataFrame(iris_df.data, columns=iris_df.feature_names)

print(iris_df.head())

iris_df = (iris_df - iris_df.mean()) / iris_df.std()

x = iris_df['data']
y = iris_df['target']

encoder = OneHotEncoder()
y = encoder.fit_transform(y.reshape(-1, 1))

test_size = 0.2

X_train, X_test, y_train, y_test = train_test_split(x, y,
                                                    test_size=test_size,
                                                    shuffle=True,
                                                    random_state=13)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(10, input_shape=(4,), activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

