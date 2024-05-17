
# 

import pandas as pd
import numpy as np
import xlwings as xl
import time

st = time.time()

data = xl.load(index = 1)
df = data.copy()
knn_df = []
listt = []

print (data.shape[0])

def dist_matrix(df0):
    init = time.time()
    # Convert DataFrame to NumPy array
    data_array = df0.to_numpy()
    
    # Calculate pairwise distances using broadcasting
    distances = np.linalg.norm(data_array[:, np.newaxis, :] - data_array[np.newaxis, :, :], axis=-1)
    
    endd = time.time()
    print(f"{endd - init}___construir matrix de distancias")
    
    return distances

def main(records):
    init = time.time()
    listt = []

    while np.any(records > 0):
        indices = np.argwhere(records > 0)
        min_position = indices[records[indices[:, 0], indices[:, 1]].argmin()]
        rec1, rec2 = min_position

        listt.append([rec1, rec2])
        records[rec1] = 0
        records[rec2] = 0
        records[:, rec1] = 0
        records[:, rec2] = 0
    endd = time.time()
    print(str(endd - init) + '___' + 'construir la relaccion')
    return listt

distances = dist_matrix(df)

pairs = main(distances)

pairs_final = []

for i in range(len(pairs)):    
    pair = [df.index[pairs[i][0]], df.index[pairs[i][1]]]
    pairs_final.append(pair)

pairs_final = pd.DataFrame(pairs_final)

pairs_final.to_clipboard()

'''
df.index[2]


pairs_final


print(pairs)
pairs[5][1]

print (len(pairs))
nd = time.time()
print (nd - st)

6+6


import numpy as np

listt = []

# Create a 2D numpy array
arr = np.array([[0,158.99,90.28,59.15,174.31,327.17,169.72],
    [158.99,0,77.26,181.22,54.59,180.9,34.55],
    [90.28,77.26,0,129.99,84.08,236.95,80.52],
    [59.15,181.22,129.99,0,210.64,359.79,201.02],
    [174.31,54.59,84.08,210.64,0,153.21,22.32],
    [327.17,180.9,236.95,359.79,153.21,0,159.04],
    [169.72,34.55,80.52,201.02,22.32,159.04,0]])

while len(arr[np.where(arr>0)])>0:
    indices = np.where(arr>0)

    min_value = np.min(arr[indices])
    min_position = np.where(arr == min_value)

    rec1 = min_position[0][0]
    rec2 = min_position[1][0]

    arr[rec1][rec2]=0
    arr[rec2][rec1]=0

    listt.append([rec1, rec2])

    arr



[[0,158.99,90.28,59.15,174.31,327.17,169.72],
[158.99,0,77.26,181.22,54.59,180.9,34.55],
[90.28,77.26,0,129.99,84.08,236.95,80.52],
[59.15,181.22,129.99,0,210.64,359.79,201.02],
[174.31,54.59,84.08,210.64,0,153.21,22.32],
[327.17,180.9,236.95,359.79,153.21,0,159.04],
[169.72,34.55,80.52,201.02,22.32,159.04,0]]
'''
