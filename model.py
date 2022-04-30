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
        # self.gru2 = tf.keras.layers.GRU(rnn_units,
        #                         return_sequences=True,
        #                         return_state=True)

        self.dense = tf.keras.layers.Dense(vocab_size)
        self.softmax = tf.keras.layers.Softmax(axis=-1)

    # @tf.function
    def call(self, inputs, states=None, return_state=False, training=True):
        x = inputs
        x_notes = self.embedding_notes(x[:,:,0], training=training)
        x_timing = self.embedding_time(x[:,:,1], training=training)
        x = tf.concat([x_notes, x_timing], axis=2)
        x = tf.reshape(x, shape=tf.stack([tf.shape(x)[0], tf.shape(x)[1], -1]))

        if states is None:
            states = self.gru.get_initial_state(x)
        
        x, states = self.gru(x, initial_state=states, training=training)
        x, states = self.gru1(x, initial_state=states, training=training)
        # x, states = self.gru2(x, initial_state=states, training=training)
        
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x


class OneStep(tf.keras.Model):
  def __init__(self, model, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model

  def generate_one_step(self, inputs, states=None):

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, states = self.model(inputs=inputs, states=states,
                                          return_state=True, training=False)
    # Only use the last prediction.
    predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)

    # Return the characters and model state.
    return predicted_ids, states