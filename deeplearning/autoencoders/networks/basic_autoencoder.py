from keras.layers import Input, Dense
from keras.models import Model
from keras.models import Sequential
from keras import regularizers
from keras.layers import advanced_activations
import keras.backend as K


class bA:
    @staticmethod
    def build(enc_dim, data_dim):

        model = Sequential()

        # enc_dim is the dimension of the hidden_layer
        model.add(Dense(enc_dim, activation="relu",
                        input_shape=(data_dim, )))

        model.add(Dense(enc_dim, activation="sigmoid"))

        return model

    @staticmethod
    def build_with_model(enc_dim, data_dim):

        input = Input(shape=(data_dim, ))
        # "encoded" is the encoded representation of the input
        encoded = Dense(enc_dim, activation='relu')(input)
        # encoded = Dense(enc_dim, activation='relu',
        #                 W_regularizer=regularizers.l2(10e-1))(input)
        # "decoded" is the lossy reconstruction of the input
        #decoded = Dense(data_dim, activation='sigmoid')(encoded) #to kanw linear katw giati 8a allaksw to normalization kai 8a xw nagative values se joints (tetartimoria)
        decoded = Dense(data_dim, activation='linear')(encoded)
        # decoded = Dense(data_dim, activation='sigmoid',
        #                         W_regularizer=regularizers.l2(0.1))(encoded)


        # this model maps an input to its reconstruction
        autoencoder = Model(input=input, output=decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input=input, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(enc_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        return autoencoder, encoder, decoder

    @staticmethod
    def build_with_model_2d(enc_dim, data_dimx, data_dimy):

        input = Input(shape=(data_dimx, data_dimy, ))
        # "encoded" is the encoded representation of the input
        encoded = Dense(enc_dim, activation='relu')(input)
        # encoded = Dense(enc_dim, activation='relu',
        #                 W_regularizer=regularizers.l2(10e-1))(input)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(data_dimx, data_dimy, activation='sigmoid')(encoded)


        # this model maps an input to its reconstruction
        autoencoder = Model(input=input, output=decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input=input, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(enc_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        return autoencoder, encoder, decoder

    @staticmethod
    def build_with_model_elu(enc_dim, data_dim):
        elu = advanced_activations.ELU(alpha=1.0)

        input = Input(shape=(data_dim, ))
        # "encoded" is the encoded representation of the input
        encoded = Dense(enc_dim, activation=elu)(input)
        # "decoded" is the lossy reconstruction of the input
        decoded = Dense(data_dim, activation='sigmoid')(encoded)


        # this model maps an input to its reconstruction
        autoencoder = Model(input=input, output=decoded)

        # this model maps an input to its encoded representation
        encoder = Model(input=input, output=encoded)

        # create a placeholder for an encoded (32-dimensional) input
        encoded_input = Input(shape=(enc_dim,))
        # retrieve the last layer of the autoencoder model
        decoder_layer = autoencoder.layers[-1]
        # create the decoder model
        decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

        return autoencoder, encoder, decoder





