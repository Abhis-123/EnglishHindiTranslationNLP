import os
from helper import load_variable, save_variable
import tensorflow as tf
from config import Weights_DIR
from dataset import prepare_data
from models.lstm.model import LSTMModel
max_sentence_length = 16
import numpy as np

class Translate:
    def __init__(self, path_to_weights=Weights_DIR, data=None, max_sentence_length=16, embedding_size=256):
        if path_to_weights is not None:
            self.path_to_weights = path_to_weights
        
        if data is not None:
            self.data = data
        else:
            print(
                "to instantiate the translation pass the path to weights or pass the training data")
            assert False
        self.en_hi_model = None
        self.hi_en_model = None
        self.english_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', oov_token='<OOV>', lower=False)
        self.hindi_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters='', oov_token='<OOV>', lower=False)
        self.english_vocab_size = None
        self.hindi_vocab_size = None
        self.validation_split = 0.15
        self.max_sentence_length = max_sentence_length

    def load_weights(self):
        if os.path.exists(os.path.join(self.path_to_weights, 'english_tokenizer')):
            self.english_tokenizer = load_variable(
                os.path.join(self.path_to_weights, 'english_tokenizer'))
            self.english_vocab_size = len(
                self.english_tokenizer.word_index) + 1
        else:
            print('Could not find english tokenizer weights')

        if os.path.exists(os.path.join(self.path_to_weights, 'hindi_tokenizer')):
            self.hindi_tokenizer = load_variable(
                os.path.join(self.path_to_weights, 'hindi_tokenizer'))
            self.hindi_vocab_size = len(self.hindi_tokenizer.word_index)+1
        else:
            print("Could not find hindi_tokenizer weights")

        if os.path.exists(os.path.join(self.path_to_weights, 'hi_en_model.h5')):
            self.hi_en_model = LSTMModel(encoder_vocab_size=self.hindi_vocab_size, decoder_vocab_size=self.english_vocab_size)
            self.hi_en_model(np.array([[[1]*self.max_sentence_length], [[1]*self.max_sentence_length]]))
            self.hi_en_model.load_weights(os.path.join(self.path_to_weights, 'hi_en_model.h5'))
        else:
            print("could not load hi_en_model")

        if os.path.exists(os.path.join(self.path_to_weights, 'en_hi_model.h5')):
            self.en_hi_model = LSTMModel(encoder_vocab_size=self.english_vocab_size, decoder_vocab_size=self.hindi_vocab_size)
            self.en_hi_model(np.array([[[1]*self.max_sentence_length], [[1]*self.max_sentence_length]]))
            self.en_hi_model.load_weights(os.path.join(self.path_to_weights, 'en_hi_model.h5'))
        else:
            print('Could not load en_hi_model')

    def tokenize_data(self):
        print("filtering the sentences on the basis of their length")
        data = self.data
        data = data[data['en'].apply(
            lambda x: len(x.split()) < max_sentence_length)]
        data = data[data['hi'].apply(
            lambda x: len(x.split()) < max_sentence_length)]
        print(
            f"{data.__len__()} sentence pairs are valid or less than max_sentence_length")
        english_sentences = self.data['en'].to_list()
        hindi_sentences = self.data['hi'].to_list()

        print("start tokenization..")
        self.english_tokenizer.fit_on_texts(english_sentences)
        self.hindi_tokenizer.fit_on_texts(hindi_sentences)
        self.english_sequences = self.english_tokenizer.texts_to_sequences(
            english_sentences)
        self.hindi_sequences = self.hindi_tokenizer.texts_to_sequences(
            hindi_sentences)

        self.english_vocab_size = len(self.english_tokenizer.word_index) + 1
        self.hindi_vocab_size = len(self.hindi_tokenizer.word_index) + 1
        print("English Vocab Size: ", self.english_vocab_size)
        print("Hindi Vocab Size: ", self.hindi_vocab_size)

    def train_en_hi(self, num_epochs=10, optimizer='rmsprop', metrics=['accuracy'], batch_size= None , **kwargs):

        if self.en_hi_model == None:
            self.en_hi_model = LSTMModel(encoder_vocab_size=self.english_vocab_size,
                                         decoder_vocab_size=self.hindi_vocab_size)
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(self.english_sequences,
                                                                       maxlen=max_sentence_length,
                                                                       padding='post')

        decoder_inputs = []
        decoder_outputs = []
        for sentence in self.hindi_sequences:
            decoder_inputs.append(sentence[:-1])
            decoder_outputs.append(sentence[1:])
        decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_inputs,
                                                                       maxlen=max_sentence_length,
                                                                       padding='post')
        decoder_outputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_outputs,
                                                                        maxlen=max_sentence_length,
                                                                        padding='post')
        self.en_hi_model.compile(optimizer=optimizer,
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=metrics)

        callback1 = tf.keras.callbacks.ModelCheckpoint(
            filepath=Weights_DIR+"\\model\\",
            monitor='val_accuracy',
            mode='max'
        )
        self.en_hi_model.fit([encoder_inputs, decoder_inputs],
                             decoder_outputs,epochs=num_epochs , batch_size= batch_size,validation_split=self.validation_split,
                             callbacks=[callback1], **kwargs)

    def train_hi_en(self, num_epochs=10, optimizer='rmsprop', metrics=['accuracy'], batch_size=None, **kwargs):
        if self.hi_en_model == None:
            self.hi_en_model = LSTMModel(encoder_vocab_size=self.hindi_vocab_size,
                                         decoder_vocab_size=self.english_vocab_size)
        encoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(self.hindi_sequences,
                                                                       maxlen=max_sentence_length,
                                                                       padding='post')

        decoder_inputs = []
        decoder_outputs = []
        for sentence in self.english_sequences:
            decoder_inputs.append(sentence[:-1])
            decoder_outputs.append(sentence[1:])
        decoder_inputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_inputs,
                                                                       maxlen=max_sentence_length,
                                                                       padding='post')
        decoder_outputs = tf.keras.preprocessing.sequence.pad_sequences(decoder_outputs,
                                                                        maxlen=max_sentence_length,
                                                                        padding='post')
        self.hi_en_model.compile(optimizer=optimizer,
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                 metrics=metrics)

        callback1 = tf.keras.callbacks.ModelCheckpoint(
            filepath=Weights_DIR+"\\model\\",
            monitor='val_accuracy',
            mode='max'
        )
        self.hi_en_model.fit([encoder_inputs, decoder_inputs],
                             decoder_outputs, epochs= num_epochs, batch_size= batch_size, validation_split=self.validation_split,
                             callbacks=[callback1], **kwargs)
                             
    def train(self, model_to_train=None, num_epochs=2, optimizer='rmsprop', metrics=['accuracy'],batch_size=16, **kwargs):
        self.tokenize_data()
        if model_to_train == 'en_hi':
            self.train_en_hi(num_epochs=num_epochs,
                             optimizer=optimizer, metrics=metrics,batch_size= batch_size, **kwargs)
        elif model_to_train == 'hi_en':
            self.train_hi_en(num_epochs=num_epochs,
                             optimizer=optimizer, metrics=metrics,batch_size= batch_size, **kwargs)
        else:
            self.train_hi_en(num_epochs=num_epochs,
                             optimizer=optimizer, metrics=metrics,batch_size= batch_size, **kwargs)
            self.train_en_hi(num_epochs=num_epochs,
                             optimizer=optimizer, metrics=metrics,batch_size= batch_size, **kwargs)

    def save_model(self, path=None):
        if path != None and os.path.exists(path):
            self.path_to_weights = path
        print(f"saving model at {self.path_to_weights}")

        save_variable(self.english_tokenizer, os.path.join(
            self.path_to_weights + "\\english_tokenizer"))
        save_variable(self.hindi_tokenizer, os.path.join(
            self.path_to_weights + "\\hindi_tokenizer"))
        self.en_hi_model.save_weights(os.path.join(self.path_to_weights + "\\en_hi_model.h5"))
        self.hi_en_model.save_weights(os.path.join(self.path_to_weights + "\\hi_en_model.h5"))



    def translate_sentence_to_hindi(self,sentence):
        if self.english_tokenizer ==None or self.hindi_tokenizer==None or self.en_hi_model==None:
            print("the translate object is not initialized properly")
            return sentence
        
        return self.en_hi_model.predict_sequence(sentence, self.english_tokenizer, self.hindi_tokenizer)


    def translate_sentence_to_english(self,sentence):
        if self.english_tokenizer ==None or self.hindi_tokenizer==None or self.hi_en_model==None:
            print("the translate object is not initialized properly")
            return sentence
        return self.hi_en_model.predict_sequence(sentence, self.hindi_tokenizer, self.english_tokenizer)

if __name__ == "__main__":
    train_df = prepare_data(type='train', max_entries=50000)
    trans = Translate(path_to_weights=Weights_DIR, data=train_df)
    trans.train(num_epochs=10)
    trans.save_model()
    trans1 = Translate(path_to_weights=Weights_DIR, data=train_df)
    trans1.load_weights()
    en ='give your application an accessibility workout ' 
    print(en)
    hi = trans1.translate_sentence_to_hindi(en)
    print(hi)
    en = trans1.translate_sentence_to_english(hi)
    print(en)