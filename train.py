
import time
import tensorflow as tf
from model import MusicRNN
import os


def train(training_data, config) -> MusicRNN:
  """
  trains the RNN

  Args:
      training_data tf.Data.dataset: tf.
      config dictionary: _description_

  Returns:
      MusicRNN: fully trained RNN
  """

  # for some reason CPU is faster to train
  os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  model = MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units)

  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,
    decay_steps=5000,
    decay_rate=0.9)

  optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule)

  # models[i].compile(optimizer='adam', loss=loss)
  model.compile(optimizer=optimizer, loss=loss)

  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt") # _{epoch}
  checkpoint = tf.train.Checkpoint(model=model) # optimizer='RMSprop',

  mean = tf.metrics.Mean()

  for input_example_batch, target_example_batch in training_data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  # model.load_weights(checkpoint_dir + "checkpoint_29.index")
  # training loop
  for epoch in range(config.epochs):
      start = time.time()

      mean.reset_states()
      for (batch_n, (inp, target)) in enumerate(training_data):
        
        logs = model.train_step([inp, target[:,:,0]])
        mean.update_state(logs['loss'])

        if batch_n % 20 == 0:
          template = f"Epoch {epoch+1} Batch {batch_n} Loss {logs['loss']:.4f}"
          print(template)

      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
          # model.save_weights(checkpoint_prefix.format(epoch=epoch))
          checkpoint.save(checkpoint_prefix)


      print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_"*80)

  checkpoint.save(f"{checkpoint_prefix}")

  return model



def load_model(config, training_data)->MusicRNN:

  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  
  # make model
  model = MusicRNN(
      vocab_size=config.vocab_size,
      embedding_dim=config.embedding_dim,
      rnn_units=config.rnn_units)

  model.compile(optimizer='RMSprop', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

  # flush inputs through to make the graph
  for input_example_batch, target_example_batch in training_data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

  # load weights
  # model.load_weights(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  # status.assert_consumed()

  return model
