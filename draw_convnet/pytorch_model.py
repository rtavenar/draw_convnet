from numpy.lib.arraysetops import isin
import torch.nn as nn


def read_model(input_image, pytorch_model):
    """Assumes the model is made of input, conv2d, maxpool2d, avgpool2d
    flatten and linear layers only."""
    input_image_channel_last = input_image.permute(1, 2, 0)  # (H, W, C)
    input_image = input_image.unsqueeze(0)  # (N==1, C, H, W)
    model_description = {}
    model_description["input_shape"] = input_image_channel_last.shape
    model_description["layers"] = []
    for layer in pytorch_model:
        output = layer(input_image)
        input_image = output
        patch_size = None
        if isinstance(layer, nn.Conv2d):
            output_shape = output.squeeze().permute(1, 2, 0).shape
            below_string = f"Convolution\n{layer.kernel_size[0]}x{layer.kernel_size[1]} kernel"
            patch_size = layer.kernel_size
        elif isinstance(layer, nn.MaxPool2d):
            output_shape = output.squeeze().permute(1, 2, 0).shape
            if isinstance(layer.kernel_size, int):
                layer.kernel_size = (layer.kernel_size, layer.kernel_size)
            below_string = f"Max-Pooling\n{layer.kernel_size[0]}x{layer.kernel_size[1]} kernel"
            patch_size = layer.kernel_size
        elif isinstance(layer, nn.AvgPool2d):
            output_shape = output.squeeze().permute(1, 2, 0).shape
            if isinstance(layer.kernel_size, int):
                layer.kernel_size = (layer.kernel_size, layer.kernel_size)
            below_string = f"Avg-Pooling\n{layer.kernel_size[0]}x{layer.kernel_size[1]} kernel"
            patch_size = layer.kernel_size
        elif isinstance(layer, nn.Flatten):
            output_shape = (output.shape[1], )
            below_string = f"Flatten"
        elif isinstance(layer, nn.Linear):
            output_shape = (output.shape[1], )
            below_string = f"Fully\nconnected"
        else:
            output_shape = output.shape
            below_string = layer.__class__.__name__
        model_description["layers"].append(
            (below_string, output_shape, patch_size)
        )
    return model_description


if __name__ == "__main__":
    from pprint import pprint
    import torch

    input_image = torch.randn(3, 32, 32)  # (C, H, W)
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="valid"),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, padding="same"),
        nn.AvgPool2d(kernel_size=(2)),
        nn.Flatten(),  # if image shape==(3, 32, 32), Flatten gives (1, 1568)
        nn.Linear(in_features=1568, out_features=124),
        nn.Linear(in_features=124, out_features=10),
    )

    pprint(read_model(input_image, model))
