import numpy as np
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input


def extract_features(image):
    # load the model
    model = VGG16()
    # re-structure the model
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature


def getdistance(widget_list):
    distance_list = []
    for group in widget_list:
        img_np1 = group[0]
        img_np2 = group[1]
        # get image feature
        feature_1 = extract_features(img_np1)
        feature_2 = extract_features(img_np2)
        distance = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        distance_list.append(distance)
    return distance_list
