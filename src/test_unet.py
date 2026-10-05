import torch

from unet import UNet


print("================================")
print("GeoVision AI U-Net Test")
print("================================")


# Create model
model = UNet(
    in_channels=6,
    num_classes=5
)


print("Model created successfully.")


# Fake Sentinel-2 batch
x = torch.randn(
    4,
    6,
    256,
    256
)


print("\nInput:")
print(x.shape)


# Forward pass
with torch.no_grad():

    output = model(x)


print("\nOutput:")
print(output.shape)


print("\nExpected:")
print("Input : [4, 6, 256, 256]")
print("Output: [4, 5, 256, 256]")


# Parameter count
parameters = sum(
    p.numel()
    for p in model.parameters()
)


print("\nTotal parameters:")
print(f"{parameters:,}")


print("\nU-Net test complete!")