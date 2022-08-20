import tensorflow as tf
import numpy as np
class LSTMModel(tf.keras.Model):

    def __init__(self,encoder_vocab_size = None, decoder_vocab_size = None, embedding_size = 128, num_rnn_units=32,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_vocab_size = encoder_vocab_size
        self.decoder_vocab_size = decoder_vocab_size
        self.embedding_size     = embedding_size
        # encoder
        self.input_1      = tf.keras.layers.InputLayer(input_shape=(None,), name = 'input_1')
        self.embedding_1  =   tf.keras.layers.Embedding(encoder_vocab_size, embedding_size,mask_zero=True, name = 'embedding_1')
        self.encoder_lstm =   tf.keras.layers.LSTM(num_rnn_units, return_state=True,  name ='encoder_lstm' )
        # decoder 
        self.input_2      = tf.keras.layers.InputLayer(input_shape=(None,), name='input_2')
        self.embedding_2  = tf.keras.layers.Embedding(decoder_vocab_size, embedding_size,mask_zero=True,name= "embedding_2")
        self.decoder_lstm = tf.keras.layers.LSTM(num_rnn_units,activation='relu',return_sequences=True, return_state=True, name ='decoder_lstm' )

        self.token_layer = tf.keras.layers.Dense(decoder_vocab_size,activation='softmax', name = 'token_layer')

    def call(self,inputs):
        encoder_input = self.input_1(inputs[0])
        decoder_input = self.input_2(inputs[1])
        # encode the inputs 
        encoder_embed = self.embedding_1(encoder_input)
        # run rnn on the encoded sequence
        _, state_h, state_c = self.encoder_lstm(encoder_embed)
        # decode the target 
        decoder_embed = self.embedding_2(decoder_input)
        x, _,_ = self.decoder_lstm(decoder_embed, initial_state=[state_h, state_c])
        return self.token_layer(x)
  
    # def get_config(self):
    #   config = super.get_config()
    #   config['encoder_vocab_size'] = self.encoder_vocab_size
    #   config['decoder_vocab_size'] = self.decoder_vocab_size
    #   config['embedding_size']     = self.embedding_size
    #   return config
    def predict_sequence(self,text, input_tokenizer, output_tokenizer, max_len= 16):
      if type(text)!=list:
        text = [text]
      input_sequence = input_tokenizer.texts_to_sequences(text)
      if type(input_sequence)==list:
        input_sequence = np.array(input_sequence)
      encoder_embed = self.embedding_1(input_sequence)
        # run rnn on the encoded sequence
      _, next_h, next_c = self.encoder_lstm(encoder_embed)
      curr_token = [[0]]
      curr_token[0][0] = output_tokenizer.word_index['<START>']

      out_seq = ""
      for i in range(max_len):
        decoder_embedding = self.embedding_2(np.array(curr_token))
        x, next_h, next_c = self.decoder_lstm(decoder_embedding, initial_state=[next_h, next_c])
        x = self.token_layer(x)
        next_token = np.argmax(x[0,0,:])
        next_word = output_tokenizer.index_word[next_token]
        if next_word =="<END>":
          break
        curr_token[0][0] = next_token
        #curr_token[0].append(next_token)
        out_seq= out_seq+" "+ next_word
      return out_seq