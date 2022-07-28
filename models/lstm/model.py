
import tensorflow as tf

class LSTMModel(tf.keras.Model):

    def __init__(self,encoder_vocab_size = None, decoder_vocab_size = None, embedding_size = 256,*args, **kwargs):
        super().__init__(*args, **kwargs)

        # encoder
        self.encoder = tf.keras.Sequential(layers=[tf.keras.layers.Input(shape=(None,)),
                                        tf.keras.layers.Embedding(encoder_vocab_size, embedding_size,mask_zero=True, ),
                                        ]) 
        self.encoder_lstm =  tf.keras.layers.LSTM(embedding_size, return_state=True)
        # decoder 
        self.decoder  = tf.keras.Sequential(layers=[tf.keras.layers.Input(shape=(None,)),
                                        tf.keras.layers.Embedding(decoder_vocab_size, embedding_size,mask_zero=True, ),
                                        ])
        self.decoder_lstm = tf.keras.layers.LSTM(embedding_size,activation='relu',return_sequences=True, return_state=True)

        self.token_layer = tf.keras.layers.Dense(decoder_vocab_size,activation='softmax')

    def call(self,inputs):
        encoder_input = inputs[0]
        decoder_input = inputs[1]
        # encode the inputs 
        encoder_embed = self.encoder(encoder_input)
        # run rnn on the encoded sequence
        _, state_h, state_c = self.encoder_lstm(encoder_embed)
        # decode the target 
        decoder_embed = self.decoder(decoder_input)
        x, _,_ = self.decoder_lstm(decoder_embed, initial_state=[state_h, state_c])
        return self.token_layer(x)

    