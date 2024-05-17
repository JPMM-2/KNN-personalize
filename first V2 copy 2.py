
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
[59.15,181.22,11.02,22.32,159.04,0]] haha
'''
