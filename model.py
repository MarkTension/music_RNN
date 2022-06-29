import tensorflow as tf
from keras.layers import Layer
import keras.backend as K


class MusicRNN(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        keep_prob = 0.3
        self.embedding_notes = tf.keras.layers.Embedding(vocab_size, embedding_dim) # , input_length=16

        self.dropout1 = tf.keras.layers.Dropout(keep_prob)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=True, # True
                                    return_state=True)
        self.dropout2 = tf.keras.layers.Dropout(keep_prob)
        self.dropout2s = tf.keras.layers.Dropout(keep_prob)

        self.gru1 = tf.keras.layers.GRU(rnn_units,
                                return_sequences=True, # True
                                return_state=True)
        self.dropout3 = tf.keras.layers.Dropout(keep_prob)

        self.gru2 = tf.keras.layers.GRU(rnn_units,
                                return_sequences=False,
                                return_state=True)

        self.dropout4 = tf.keras.layers.Dropout(keep_prob)
        # self.dense = tf.keras.layers.Dense(128) # 256
        self.dense_out = tf.keras.layers.Dense(vocab_size + 2) # + 2 is for the note on or off
        self.attention = attention()

    # @tf.function
    def call(self, inputs, states=None, return_state=False, training=True):

        x = inputs
        x_notes = self.embedding_notes(x[:,:,0], training=training) #tf.expand_dims(tf.cast(x[:,:,0] / 86, tf.float32), axis=2)  #self.embedding_notes(x[:,:,0], training=training)
        x_timing = tf.expand_dims(x[:,:,1], axis=2) 
        x_on_off = tf.expand_dims(x[:,:,2], axis=2) 
        x = tf.concat([x_notes, x_timing, x_on_off], axis=2)
        
        x_attention, alpha = self.attention(x)
        x = tf.concat([x, tf.expand_dims(x_attention,axis=2)], axis=2)

        if states is None:
            states = self.gru.get_initial_state(x)
        
        if (training): x = self.dropout1(x)
        x, states = self.gru(x, initial_state=states, training=training)
        if (training): x = self.dropout2(x)

        # x_attention, alpha = attention()(x)
        # x = tf.concat([x, tf.expand_dims(x_attention,axis=2)], axis=2)

        x, states = self.gru1(x, initial_state=states, training=training)
        if (training): x = self.dropout3(x)

        # x_attention, alpha = attention()(x)
        # x = tf.concat([x, tf.expand_dims(x_attention,axis=2)], axis=2)

        x, states = self.gru2(x, initial_state=states, training=training)
        if (training): x = self.dropout4(x)

        x = tf.reshape(x, shape=[x.shape[0], -1])

        x = self.dense_out(x, training=training)

        if return_state:
            return x[:,:-2], x[:,-2:], states, alpha
        else:
            return x[:,:-2], x[:,-2:],



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
        