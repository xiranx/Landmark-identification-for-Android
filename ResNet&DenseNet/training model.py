from keras.applications.densenet import DenseNet201, preprocess_input
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Input, Lambda, Dense
from keras.models import Model, model_from_json
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
from sklearn.model_selection import train_test_split
import os
import numpy as np

# import ImageProcess

DENSENET = 1
RESNET = 2


def GetTrainData():
    return np.load("train_data.npy")


def GetTrainLabel():
    return np.load("train_labels.npy")


def GetClassification():
    return np.size(np.load("train_labels.npy"), 1)


def add_new_last_layer(base_model, nb_classes):
    x = base_model.output
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model


def create_model(type=DENSENET, save=True):
    height = 224
    x = Input(shape=(height, height, 3))
    x = Lambda(preprocess_input)(x)

    nb_classes = GetClassification()
    # print(nb_classes)
    if type == DENSENET:
        model = DenseNet201(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
    else:
        model = ResNet50(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
    model = add_new_last_layer(model, nb_classes)

    if save:
        json_model = model.to_json()
        if type == DENSENET:
            json_file = open('densenet.json', 'w')
        else:
            json_file = open('resnet.json', 'w')
        json_file.write(json_model)
        json_file.close()
    return model


def load_model(type=DENSENET):
    if type == DENSENET:
        json_file = open('densenet.json', 'r')
    else:
        json_file = open('resnet.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)

    return model


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    mod = DENSENET
    model = create_model(mod, False)
    # model = load_model(mod)

    model.compile(optimizer=SGD(lr=0.001, momentum=0.9, decay=0.0001, nesterov=True), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    if mod == DENSENET:
        filename = 'parameters/densenet0601-{epoch:02d}e-val_acc_{val_acc:.2f}.hdf5'
    else:
        filename = 'parameters/resnet0601-e{epoch:02d}-acc{val_acc:.2f}.hdf5'

    mc = ModelCheckpoint(filename, monitor='val_acc', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_loss', patience=3)
    csv = CSVLogger("DESNSENET 24 0.001.csv")

    train = GetTrainData()
    labels = GetTrainLabel()
    X_train, X_val, y_train, y_val = train_test_split(train, labels, shuffle=True, test_size=0.2, random_state=42)

    history = model.fit(x=X_train, y=y_train, batch_size=24, epochs=20, validation_data=(X_val, y_val),
                        callbacks=[es, csv])

    model.save('my_test_model.h5')
    