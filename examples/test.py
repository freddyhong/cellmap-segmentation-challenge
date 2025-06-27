import zarr
import numpy as np

arr = zarr.open("../data/predictions/jrc_cos7-1b_3D.zarr/crop235/mito/s0", mode="r")
data = arr[:]
print("Shape:", data.shape)
print("Min:", np.min(data), "Max:", np.max(data), "Mean:", np.mean(data))
print("Unique values:", np.unique(data))
