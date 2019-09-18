import tensorflow as tf
import numpy as np
from tensorflow_core.python import keras
from tqdm import tqdm

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

model.compile(optimizer='sgd', loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

with tqdm(total=500, unit='epoch') as t:
    def cbk(epoch, logs):
        t.set_postfix(logs, refresh=False)
        t.update()


    cbkWrapped = keras.callbacks.LambdaCallback(on_epoch_end=cbk)
    model.fit(xs, ys, epochs=t.total, verbose=0, callbacks=[cbkWrapped])

print(model.predict([10.0])[0][0])  # should be 31
