import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils


def run():
    start = time.time()
    with tf.device('/gpu:0'):
        nb_classes = 10

        # the data, shuffled and split between tran and test sets
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        print("X_train original shape", X_train.shape)
        print("y_train original shape", y_train.shape)

        #for i in range(9):
        #    plt.subplot(3,3,i+1)
        #    plt.imshow(X_train[i], cmap='gray', interpolation='none')
        #    plt.title("Class {}".format(y_train[i]))
        #plt.show()

        X_train = X_train.reshape(60000, 1, 28, 28)
        X_test = X_test.reshape(10000, 1, 28, 28)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train /= 255
        X_test /= 255
        print("Training matrix shape", X_train.shape)
        print("Testing matrix shape", X_test.shape)

        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        print("Training result shape", Y_train.shape)
        print("Testing result shape", Y_test.shape)


        model = Sequential()
        model.add(Convolution2D(32, nb_row=1, nb_col=1, input_shape=(1, 28, 28,)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Convolution2D(32, 1, 1))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 1)))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))  # Dropout helps protect the model from memorizing or "overfitting" the training data
        model.add(Dense(10))
        model.add(Activation('softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
        #model.compile(loss='categorical_crossentropy', optimizer='adam')

        step_size = 1000

        for i in range(0, 60000, step_size):
            sub_x = X_train[i:(step_size+i)]
            sub_y = Y_train[i:(step_size+i)]
            sub_x = sub_x.reshape(step_size, 1, 28, 28)
            sub_y = sub_y.reshape(step_size, 10)
            model.fit(sub_x, sub_y, nb_epoch=1, verbose=0)

        # model.fit(X_train, Y_train,
        #           batch_size=128, nb_epoch=1, verbose=0,
        #           validation_data=(X_test, Y_test))

        score = model.evaluate(X_test, Y_test, verbose=1)
        print('\n')
        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        # The predict_classes function outputs the highest probability class
        # according to the trained classifier for each input example.
        predicted_classes = model.predict_classes(X_test)

        # Check which items we got right / wrong
        correct_indices = np.nonzero(predicted_classes == y_test)[0]
        incorrect_indices = np.nonzero(predicted_classes != y_test)[0]
        print('\n')
        print('correct_indices count:  ', len(correct_indices))
        print('incorrect_indices count:  ', len(incorrect_indices))


    end = time.time()
    print("\nCompute Time:  ", end - start)

if __name__ == '__main__':
    run()