import numpy as np
from keras.preprocessing.image import img_to_array, load_img
import os, random
from googleapiclient import discovery
from google.cloud import error_reporting
import json
import sys
sys.path.append(".")
sys.path.append("..")


def get_random_image(base_path):
    random_image =  random.choice(os.listdir(base_path))
    image_path = os.path.join(base_path, random_image)
    return image_path

def predict(model_name, project, data, model_version='v2'):
    """
    Makes API call to AI Platform and returns prediction.
    :param model_name: REQUIRED. STRING. Name of model on AI Platform.
    :param project: your project Id.
    :param data: cleaned and preprocessed data in shape that your model expects in JSON format "{instances: [data]}"
    :param model_version: model version you want to make the request to
    :return: prediction:
    """
    project_id = f"projects/{project}"
    service = discovery.build('ml', 'v1')
    name = f"{project_id}/models/{model_name}"
    if model_version is not None:
        name += f"/versions/{model_version}"
    data_pred = json.loads(data)
    instances = data_pred['instances']

    try:
        response = service.projects().predict(
            name=name,
            body={"instances": instances}
        ).execute()

        return response['predictions']

    except Exception:
        error_client = error_reporting.Client()
        error_client.report_exception()


def get_pred_data(random_image_path):
    img_width, img_height = 28, 28
    img = load_img(random_image_path, color_mode='grayscale', target_size=(img_width,img_height))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    return x

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_prediction(data):
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    model_name = "mnist"
    project = "antsa-demo-devfest"
    data_json = json.dumps({"instances": data}, cls=NumpyEncoder)
    prediction_probs= predict(model_name, project, data_json)
    pred_list = prediction_probs[0]['output']
    prediction = {}
    for index, pred in enumerate(pred_list):
        if pred > 0.5:
            prediction['Class'] = classes[index]
            prediction['Probability'] = pred
    return prediction


def run():
    try:
        base_path = '../Data'
        random_image_path = get_random_image(base_path)
        print(random_image_path)
        x = get_pred_data(random_image_path)
        prediction = get_prediction(x)
        print(prediction)
    except Exception:
        error_client = error_reporting.Client()
        error_client.report_exception()

run()


