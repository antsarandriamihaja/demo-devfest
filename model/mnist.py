import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pysnooper
import keras_to_savedmodel
import ai_platform
import os

@pysnooper.snoop()
def get_data(num_classes):
    # input image dimensions
    img_width, img_height = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape)

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_width, img_height)
        x_test = x_test.reshape(x_test.shape[0], 1, img_width, img_height)
        input_shape = (1, img_width, img_height)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
        x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)
        input_shape = (img_width, img_height, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test, input_shape

@pysnooper.snoop()
def get_model(input_shape, num_classes, x_train, y_train, x_test, y_test, filename):
    batch_size = 128
    epochs = 3
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))
    model_json = model.to_json()

    with open(f"{filename}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f'{filename}.h5')


def run():
    num_classes = 10
    export_path = "trained_model"
    deployment_uri = "gs://antsa-demo-devfest/trained_model"
    model_name = "mnist"
    model_version = "v1"
    project_id = "antsa-demo-devfest"
    filename = "mnist_model"
    x_train, y_train, x_test, y_test, input_shape = get_data(num_classes)
    get_model(input_shape, num_classes, x_train, y_train, x_test, y_test, filename)
    keras_to_savedmodel.convert(f'{filename}.h5', f'{filename}.json', export_path)
    ai_platform.create_model_version(model_name, model_version, project_id, deployment_uri)

get_data(10)