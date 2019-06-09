from keras.optimizers import SGD
import cv2


import numpy as np


from keras.models import model_from_json
from keras.applications import imagenet_utils
from keras.applications.resnet50 import preprocess_input

EVALUATE = 1
PREDICT = 2

def GetTestData():
    return np.load("test_data.npy")

def GetTestLabel():
    return np.load("test_labels.npy")


def PredictOrEvaluate(filename,weightname,mod,imgpath=None):
    with open(filename, 'r') as file:
        loaded_model_string = file.read()
    loaded_model = model_from_json(loaded_model_string, custom_objects={"imagenet_utils": imagenet_utils})
    loaded_model.load_weights(weightname)
    loaded_model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True),
                         loss='categorical_crossentropy', metrics=['accuracy'])

    if mod == EVALUATE:
        testdata = GetTestData()
        testlabel = GetTestLabel()
        result = loaded_model.evaluate(testdata,testlabel,verbose=1)
        print(result)
    else:
        img = cv2.imread(imgpath)
        img = cv2.resize(img, (244, 244))
        image = np.expand_dims(img, axis=0)
        result = loaded_model.predict(image)
        print(result)


if __name__ == '__main__':
    mod = PREDICT
    PredictOrEvaluate('resnet_model.json','parameters/resnet-e02-acc1.00.hdf5',mod,'C:\\Users\\apple\\Desktop\\peppa.JPG')