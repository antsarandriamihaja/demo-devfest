import keras
from keras.models import load_model
import sys
sys.path.append(".")
sys.path.append("..")
from predict import get_random_image, get_pred_data
from model.keras_to_savedmodel import get_keras_model



def test_prediction():
    base_path = '../Data'
    random_image_path = get_random_image(base_path)
    print(random_image_path)
    x = get_pred_data(random_image_path)
    # print(x.shape)
    model = get_keras_model('../model/mnist_model.h5', '../model/mnist_model.json')
    preds = model.predict_classes(x)
    prob = model.predict_proba(x)
    print(preds, prob)

test_prediction()