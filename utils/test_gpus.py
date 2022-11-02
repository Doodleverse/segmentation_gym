# Written by Dr Daniel Buscombe, Marda Science LLC
# for the USGS Coastal Change Hazards Program
#
# MIT License
#
# Copyright (c) 2022, Marda Science LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os 

num_epochs = 3
USE_GPU = True
# USE_MULTI_GPU = True 
USE_MULTI_GPU = False

if USE_MULTI_GPU:
    GPU_ID = '1,2'
else:
    GPU_ID = '0'

os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

import tensorflow as tf
import tensorflow.keras as keras


# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')
# # tf.debugging.set_log_device_placement(True)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)


if physical_devices:
    try:
        tf.config.experimental.set_visible_devices(physical_devices, 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

for i in physical_devices:
    tf.config.experimental.set_memory_growth(i, True)

print(tf.config.get_visible_devices())


def get_compiled_model():
    # Make a simple 2-layer densely-connected neural network.
    inputs = keras.Input(shape=(784,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.Dense(256, activation="relu")(x)
    outputs = keras.layers.Dense(10)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
        run_eagerly=True,
        steps_per_execution=2
    )
    return model


def get_dataset():
    batch_size = 400
    num_val_samples = 2000

    # Return the MNIST dataset in the form of a [`tf.data.Dataset`](https://www.tensorflow.org/api_docs/python/tf/data/Dataset).
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Preprocess the data (these are Numpy arrays)
    x_train = x_train.reshape(-1, 784).astype("float32") / 255
    x_test = x_test.reshape(-1, 784).astype("float32") / 255
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    # Reserve num_val_samples samples for validation
    x_val = x_train[-num_val_samples:]
    y_val = y_train[-num_val_samples:]
    x_train = x_train[:-num_val_samples]
    y_train = y_train[:-num_val_samples]
    return (
        tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size),
        tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size),
    )

if USE_MULTI_GPU:
    # Create a MirroredStrategy.
    strategy = tf.distribute.MirroredStrategy([p.name.split('/physical_device:')[-1] for p in physical_devices], cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        # Everything that creates variables should be under the strategy scope.
        # In general this is only model construction & `compile()`.
        model = get_compiled_model()

else:
    model = get_compiled_model()

# Train the model on all available devices.
train_dataset, val_dataset, test_dataset = get_dataset()
model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

# Test the model on all available devices.
model.evaluate(test_dataset)