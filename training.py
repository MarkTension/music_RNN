import tensorflow as tf
from keras.layers import Dense, Flatten, Activation, RepeatVector, Permute, Multiply, Lambda, merge
from keras import backend as K


class Teacher_model(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, rnn_units):
    super().__init__(self)
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

    self.rnn_units = rnn_units
    # todo: try each voice a different embedding and separate RNN
    self.gru0 = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_dropout=0.3,
                                    dropout=0.3)
    self.gru1 = tf.keras.layers.GRU(rnn_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_dropout=0.3,
                                    dropout=0.3)


    self.gru2 = tf.keras.layers.GRU(rnn_units,
                                    return_sequences=False,
                                    return_state=True,
                                    recurrent_dropout=0.3,
                                    dropout=0.3)

    self.normalization = tf.keras.layers.BatchNormalization()
    self.dense = tf.keras.layers.Dense(vocab_size)
    self.dense_attention = tf.keras.layers.Dense(1, activation='tanh')

  def call(self, inputs, states=None, return_state=False, training=False):
    x = inputs

    print(f"input {x.shape}")
    x = self.embedding(x, training=training)
    x = self.normalization(x)
    print(f"after embedding {x.shape}")
    x = tf.reshape(x, shape=tf.stack([tf.shape(x)[0], tf.shape(x)[1], -1]))
    print(f"after reshape {x.shape}")

    if states is None:
      states = self.gru0.get_initial_state(x)

    x, states = self.gru0(x, initial_state=states, training=training) # , unroll=True
    x, states = self.gru1(x, initial_state=states, training=training) # , unroll=True
    x, states = self.gru2(x, initial_state=states, training=training) # , unroll=True

    # x = self.normalization(x)
    print(f"after GRU {x.shape}")

    # # attention part
    # e = self.dense_attention(x)
    #
    # e = Flatten()(e)
    # a = Activation('softmax')(e)
    # temp = RepeatVector(self.rnn_units)(a)
    # temp = Permute([2, 1])(temp)
    # # multiply weight with lstm layer o/p
    # output = merge.Multiply()([x, temp])
    # # Get the attention adjusted output state
    # x = Lambda(lambda values: K.sum(values, axis=1))(output)

    print(f"after attention {x.shape}")
    x = tf.reshape(x, tf.stack([tf.shape(x)[0], 4, -1]))
    print(f"after final reshape {x.shape}")
    x = self.dense(x, training=training)
    print(f"after dense {x.shape}")
    x = tf.keras.layers.Softmax(axis=-1)(x)

    if return_state:
      return x, states
    else:
      return x