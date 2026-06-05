from torchgeo.datasets import SSL4EOS12
import torch

ssl4eo = SSL4EOS12(root="../data/ssl4eo", split="s2c", download=True, checksum=True)

print(ssl4eo)

