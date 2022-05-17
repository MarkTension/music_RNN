import tensorflow as tf
from keras.layers import Layer
import keras.backend as K


class MusicRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding_notes = tf.keras.layers.Embedding(vocab_size, embedding_dim) # , input_length=16
        self.embedding_time = tf.keras.layers.Embedding(6, 3) # , input_length=16
        
        self.gru = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True,
                                    return_state=True)
        self.gru1 = tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                return_state=True)
        self.gru2 = tf.keras.layers.GRU(rnn_units,
                                return_sequences=True,
                                return_state=True)

        # self.dense = tf.keras.layers.Dense(256)
        self.dense_out = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    # @tf.function
    def call(self, inputs, states=None, return_state=False, training=True):
        x = inputs
        x_notes = self.embedding_notes(x[:,:,0], training=training)
        x_timing = tf.cast(tf.expand_dims(x[:,:,1], axis=2) , tf.float32)
        x = tf.concat([x_notes, x_timing], axis=2)

        x_attention, alpha = attention()(x)
        x = tf.concat([x, tf.expand_dims(x_attention,axis=2)], axis=2)

        if states is None:
            states = self.gru.get_initial_state(x)
        
        x, states = self.gru(x, initial_state=states, training=training)
        x, states = self.gru1(x, initial_state=states, training=training)
        x, states = self.gru2(x, initial_state=states, training=training)
        
        x = tf.reshape(x, shape=[x.shape[0], -1])
        # x = self.dense(x, training=training)

        x = self.dense_out(x, training=training)

        if return_state:
            return x, states, alpha
        else:
            return x



# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
    
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b) # TODO: add also rnn state as well with additional weight and bias
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        x = x * alpha
        context = K.sum(x, axis=2)
        return context, alpha