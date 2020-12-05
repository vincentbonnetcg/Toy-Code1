"""
@author: Vincent Bonnet
@description : Optimizer for the problem
"""
from keras.layers import Input, Dense
from keras.models import Model
import keras

class ModelOpt:

    def __init__(self):
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None

    def create_model(self, in_shape, out_shape):
        # Create NN model
        input_layer = Input(shape=(in_shape,))
        layer_0 = Dense(units=32, activation='tanh', name='HiddenLayer0')(input_layer)
        layer_1 = Dense(units=128, activation='tanh', name='HiddenLayer1')(layer_0)
        layer_2 = Dense(units=out_shape, activation='tanh', name='HiddenLayer2')(layer_1)
        self.model = Model(input_layer, layer_2)

        optimizer = keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, amsgrad=False)
        loss = keras.losses.mean_squared_error
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        #self.model.summary()

        return self.model

    def set_data(self, x, y, x_valid, y_valid):
        self.x_train = x
        self.y_train = y
        self.x_valid = x_valid
        self.y_valid = y_valid

    def fit(self, epochs=500, batch_size=10):
        self.model.fit(x=self.x_train, y=self.y_train,
                       epochs=epochs,
                       batch_size=batch_size,
                       shuffle=True,
                       validation_data=(self.x_valid,self.y_valid))

    def predict(self, x):
        return self.model.predict(x)

