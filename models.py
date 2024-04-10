from keras.src.applications import VGG16
from tensorflow.keras.models import Sequential
from keras.src.layers import Reshape, LSTM
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization,Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
from tensorflow.python.keras import regularizers
from keras.layers import SimpleRNN


# Simple CNN Model
class DeepANN():
    def simple_model(self, input_shape=(128, 128, 3), op="sgd"):  # model 1
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='sigmoid'))

        model.compile(loss="binary_crossentropy", optimizer=op, metrics=['accuracy'])

        return model

    def simple_model_multiclass(self, input_shape=(128, 128, 3), optimizer="sgd"):  # model 1
        model = Sequential()

        model.add(Flatten(input_shape=input_shape))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

        return model

    def cnn_vgg(self):   # model 7
        model = Sequential()
        model.add(VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3)))
        # fully connected layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # output layer
        model.add(Dense(3, activation="softmax"))
        # compilation
        model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model






    def cnn_model(self, input_shape=(128, 128, 3), optimizer='sgd'):
        model = Sequential()

        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D(2, 2))
        model.add(BatchNormalization())  # Corrected line
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))


        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64,kernel_regularizer=l2(0.01), activation="relu"))
        model.add(Dense(3,kernel_regularizer=l2(0.01), activation="softmax"))
        model.add(Dropout(0.2))
        model.compile(loss="binary_crossentropy", optimizer='sgd', metrics=["accuracy"])

        return model

    def create_rnn_model(self, input_shape, no_of_classes):
        model = Sequential()

        # reshape layer to flatten the input images 28,28,3
        model.add(Reshape((input_shape[0] *
                           input_shape[1], input_shape[2]), input_shape=input_shape))
        model.add(SimpleRNN(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='sgd',
                      loss='categorical_crossentropy', metrics=['accuracy'])

        return model

    def create_LSTM_rnn_model(self, input_shape, no_of_classes):
        model = Sequential()

        # reshape layer to flatten the input images 28,28,3
        model.add(Reshape((input_shape[0] * input_shape[1],input_shape[2]), input_shape=input_shape))
        model.add(LSTM(128))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    def simple_ANN(self, input_shape=(128, 128, 3), optimizer='sgd'):  # model 3
        model = Sequential()
        # add layers
        model.add(Flatten(input_shape=input_shape))  # update input_shape
        model.add(Dense(128, activation="relu"))
        model.add(Dense(64, activation="relu"))
        model.add(Dense(3, activation="sigmoid"))  # change to 1 for binary classification

        # compile model
        model.compile(loss="categorical_crossentropy", optimizer='sgd', metrics=["accuracy"])

        return model

    def cnn_add_regularize(self):    # model 6
        model = Sequential()
        # convolutional layers
        model.add(Conv2D(32, (3, 3), activation='relu',
                         input_shape=(128, 128, 3)))  # in input  shape 3 represent  coloured images
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(0.25))
        # fully connected layer
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        # output layer
        model.add(Dense(3, activation='softmax'))
        # compilation
        model.compile(loss="binary_crossentropy", optimizer="sgd", metrics=["accuracy"])
        return model

def train_model(model_instance,tr_gen,va_gen,epochs=10):
    history = model_instance.fit(tr_gen, validation_data=va_gen,epochs=epochs)
    return history

def compare_model(models,tr_gen,va_gen,tt_gen,epochs=3):  # model 2
    histories=[]
    mo = ['ann_adam', 'ann_sgd', 'ann_rmsprop']
    for model in models:
        history=train_model(model,tr_gen,va_gen,epochs=epochs)
        mo_loss, mo_acc = model.evaluate(tt_gen)
        print("the ann Architecture")
        print(model.summary())
        histories.append(history)
        print(f"test accuracy :{mo_acc}")

        #plotting
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10))

    for i, history in enumerate(histories):
        axes[0].plot(history.history['accuracy'], label=mo[i])
        axes[1].plot(history.history['loss'], label=mo[i])

    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].set_title('Model Loss Comparison')
    axes[1].set_xlabel('Epochs')
    #axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.savefig('static/images/compare.jpg')
    plt.tight_layout()
    plt.show(block=True)

def compare_model1(models,tr_gen,va_gen,epochs=5):
    histories=[]
    mo = ['ann_adam', 'ann_sgd', 'ann_rmsprop', 'cnn_adam', 'cnn_sgd', 'cnn_rmsprop']
    for model in models:
        history=train_model(model,tr_gen,va_gen,epochs=epochs)
        print("the ann Architecture")
        print(model.summary())
        histories.append(history)
        #plotting
    fig, axes = plt.subplots(nrows=2, figsize=(10, 10))
    for i, history in enumerate(histories):
        axes[0].plot(history.history['accuracy'], label=mo[i])
        axes[1].plot(history.history['loss'], label=mo[i])
    axes[0].set_title('Model Accuracy Comparison')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].set_title('Model Loss Comparison')
    axes[1].set_xlabel('Epochs')
    #axes[1].set_ylabel('Loss')
    axes[1].legend()
    plt.savefig('static\images\compare.jpg')