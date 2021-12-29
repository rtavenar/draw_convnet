import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from numpy.lib.arraysetops import isin
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AvgPool2D

def read_model(input_image, keras_model):
    """Assumes the model is made of input, conv2d, maxpool2d, avgpool2d
    flatten and dense layers only."""
    model_description = {}
    model_description["input_shape"] = input_image.input_shape[0][1:]
    model_description["layers"] = []
    for layer in keras_model.layers:
        output_shape = layer.output_shape[1:]
        patch_size = None
        if isinstance(layer, Conv2D):
            below_string = f"Convolution\n{layer.kernel_size[0]}x{layer.kernel_size[1]} kernel"
            patch_size = layer.kernel_size
        elif isinstance(layer, MaxPool2D):
            below_string = f"Max-Pooling\n{layer.pool_size[0]}x{layer.pool_size[1]} kernel"
            patch_size = layer.pool_size
        elif isinstance(layer, AvgPool2D):
            below_string = f"Avg-Pooling\n{layer.pool_size[0]}x{layer.pool_size[1]} kernel"
            patch_size = layer.pool_size
        elif isinstance(layer, Flatten):
            below_string = f"Flatten"
        elif isinstance(layer, Dense):
            below_string = f"Fully\nconnected"
        else:
            below_string = layer.__class__.__name__
        model_description["layers"].append(
            (below_string, output_shape, patch_size)
        )
    return model_description
        
if __name__ == "__main__":
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import InputLayer
    from pprint import pprint

    input_image = InputLayer((32, 32, 3))
    model = Sequential([
        input_image,
        Conv2D(filters=32, kernel_size=3, padding="valid"),
        MaxPool2D(pool_size=2),
        Conv2D(filters=32, kernel_size=5, padding="same"),
        AvgPool2D(pool_size=2),
        Flatten(),
        Dense(units=124),
        Dense(units=10)
    ])
    pprint(read_model(input_image, model))