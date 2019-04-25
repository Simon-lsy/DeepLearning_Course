import numpy as np
import keras
from keras import backend as K
from matplotlib import pyplot as plt
import pickle


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
# standardize
x_train, x_test = x_train / 255.0, x_test / 255.0
# one hot encoder
y_train = keras.utils.to_categorical(y_train, 10)
# y_train = 0.9 * y_train + 0.1 * (1 - y_train)
y_test = keras.utils.to_categorical(y_test, 10)


model = keras.models.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.ZeroPadding2D((1, 1)),
    keras.layers.MaxPool2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])


# Condition1: standard cross entropy
# Condition2: symmetrical cross entropy
# Condition3: average

def symmetrical_cross_entropy_loss(y_true, y_pred):
    y_true = 0.9 * y_true + 0.1 * (1 - y_true)
    return K.categorical_crossentropy(y_true, y_pred) + K.categorical_crossentropy(y_pred, y_true)


def average_cross_entropy_loss(y_true, y_pred):
    cross_entropy = K.categorical_crossentropy(y_true, y_pred)
    y_true = 0.9 * y_true + 0.1 * (1 - y_true)
    symmetrical_cross_entropy = symmetrical_cross_entropy_loss(y_true, y_pred)
    return 0.5 * (cross_entropy + symmetrical_cross_entropy)


mode = 'average_cross_entropy'
if __name__ == '__main__':
    if mode == 'cross_entropy':
        model.compile(optimizer=keras.optimizers.adam(lr=0.001),
                      loss="categorical_crossentropy",
                      metrics=['accuracy'])
    if mode == 'symmetrical_cross_entropy':
        model.compile(optimizer=keras.optimizers.adam(lr=0.001),
                      loss=symmetrical_cross_entropy_loss,
                      metrics=['accuracy'])
    if mode == 'average_cross_entropy':
        model.compile(optimizer=keras.optimizers.adam(lr=0.001),
                      loss=average_cross_entropy_loss,
                      metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=30, batch_size=128, validation_split=0.1)

    plt.plot(hist.epoch, hist.history['acc'], label='acc')
    plt.plot(hist.epoch, hist.history['val_acc'], label='val_acc')
    plt.legend()
    plt.savefig(mode+'.png')

    with open(mode+'.pickle', 'wb') as f:
        pickle.dump(hist.history, f, pickle.HIGHEST_PROTOCOL)

    # with open('data2.pickle', 'rb') as f:
    #     data = pickle.load(f)

    loss, accuracy = model.evaluate(x_test, y_test)
    print('accuracy', accuracy)
    # accuracy > 75%

    model.save(mode+'.h5')






