import cv2
import os
import glob
import numpy as np

height = 224
TRAIN = 1
TEST = 2
def cate(labels, n):
    result = np.zeros((len(labels), n))
    for i in range(len(labels)):
        result[i][labels[i]] = 1
    return result

def preprocess(datapath,type):
    paths = sorted(glob.glob(datapath))
    imgnumber = []
    totalnumber = 0
    for path in paths:
        totalnumber = totalnumber+len(glob.glob(path+'/*'))
    print(paths)

    data = np.zeros((totalnumber,height,height,3),dtype=np.uint8)

    label = 0
    i = 0
    labelnumber = []
    for path in paths:
        print(path)
        imgpaths = glob.glob(path+'/*')
        for imgpath in imgpaths:
            print(imgpath)
            labelnumber.append(label)
            img = cv2.imread(imgpath)
            img = cv2.resize(img,(height,height))
            data[i] = img[:,:,::-1]
            i = i+1
        label = label+1

    labels = cate(labelnumber, label)

    if type==TRAIN:
        np.save("train_labels.npy", labels)
        np.save("train_data.npy", data)
    else:
        np.save("test_labels.npy", labels)
        np.save("test_data.npy", data)


if __name__ == '__main__' :
    trainpath = 'G:/data/*'
    testpath = 'G:/testdata/*'
    preprocess(trainpath,TRAIN)
    preprocess(testpath,TEST)
