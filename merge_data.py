import numpy as np
import cv2
import scipy.io as io
import h5py
import os

os.chdir('C:\\Users\\User\\Desktop\\CarlaSimulator\\PythonClient\\img')
i = 1
j = 1
data = {}
while(1):
    try:
        a = io.loadmat('data'+str(i))
        del a['__header__']
        del a['__version__']
        del a['__globals__']
        print(len(a.keys()))
        for k in a.keys():
            data[k[0:4]+str(j)+k[k.find('_'):]] = np.asarray(a[k])
            j += 1
        i += 1
    except:
        io.savemat('merged_data',data)
        print('Data Saved')
        break
