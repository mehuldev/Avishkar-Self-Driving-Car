import numpy as np
import cv2
"""import os
#import matplotlib.pylab as plt
#data = np.load("car_data0_racetrack-angle_0_t_0.npy")
#img = Image.fromarray(data)
#plt.figure(figsize=(1,1)
direc = os.getcwd()
i =0
for file in os.listdir(direc):
    if file.endswith(".npy"):
        data = np.load(file)
        cv2.imshow('Image{}'.format(i),data)
        cv2.waitKey(0)
        print(np.size(data))
        i+=1
        cv2.destroyAllWindows()
"""

import scipy.io as io
import h5py
a = io.loadmat('.\\img\\merged_data')
#d = list(a.items())
#da = np.asarray(d[0])
#print(d)

del a['__header__']
del a['__version__']
del a['__globals__']
count = 0
for k in a.keys():
    count += 1
    l = np.asarray(a[k])
    count += 1
    cv2.imshow('IMage1',l)
    print(k[k.find('_')+1:])
    cv2.waitKey(500)
    cv2.destroyAllWindows()

print(count)

    
