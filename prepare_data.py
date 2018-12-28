from __future__ import print_function

import numpy as np
import netCDF4 as nc

import matplotlib.pyplot as plt
#%matplotlib inline


def load_data(file_path, lags, steps):
    nc_obj = nc.Dataset(file_path)
    sst = nc_obj.variables['sst'][:]
    sst_nino34 = sst[::, 72:108:, 175:255:]
    input, target = list(), list()
    for s in range(len(sst_nino34) - lags - steps + 1):
        input.append(sst_nino34[s:s + lags])
        target.append(sst_nino34[s + lags:s + lags + steps])
    return input, target


def get_batches(input, target, batch_size):
    length = len(input[0])
    for i in range(0, length, batch_size):
        input_i = input[i:i+batch_size, :, :, :]
        target_i = target[i:i+batch_size, :, :, :]
        yield (input_i, target_i)
        # batch = data[:, i:i+batch_size, :, :]			#[20,batch_size,64,64]
        # batch_x = batch[0:10]
        # batch_y = batch[10:20]
        # yield (batch_x, batch_y)
