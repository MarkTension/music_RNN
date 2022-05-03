import tensorflow as tf


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
                                
        # self.gru2 = tf.keras.layers.GRU(rnn_units,
        #                         return_sequences=True,
        #                         return_state=True)

        self.dense = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    # @tf.function
    def call(self, inputs, states=None, return_state=False, training=True):
        x = inputs
        x_notes = self.embedding_notes(x[:,:,0], training=training)
        x_timing = tf.cast(tf.expand_dims(x[:,:,1], axis=2) , tf.float32)
        x = tf.concat([x_notes, x_timing], axis=2)
        # x = tf.reshape(x, shape=tf.stack([tf.shape(x)[0], tf.shape(x)[1], -1]))

        if states is None:
            states = self.gru.get_initial_state(x)
        
        x, states = self.gru(x, initial_state=states, training=training)
        x, states = self.gru1(x, initial_state=states, training=training)
        x, states = self.gru2(x, initial_state=states, training=training)
        
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x
