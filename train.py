import time
import tensorflow as tf
from model import MusicRNN
import os
from datetime import datetime

def compile_model(config):

  model = MusicRNN(
        vocab_size=config.vocab_size,
        embedding_dim=config.embedding_dim,
        rnn_units=config.rnn_units)

  # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
  #   initial_learning_rate=config['learning_rate'],
  #   decay_steps=5000,
  #   decay_rate=1.0) # 0.9

  lr_schedule = config['learning_rate']

  loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

  # optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr_schedule, clipnorm=3)
  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=3)

  # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=config['log_dir'], histogram_freq=1)
  # , callbacks=[tensorboard_callback]
  model.compile(optimizer=optimizer, loss=loss)
  return model, loss, optimizer

@tf.function
def train_step(model, inputs, label_notes, label_timing, loss_fn, optimizer):

  with tf.GradientTape() as tape:
    out_notes, out_timing = model(inputs)
    loss_notes = loss_fn(label_notes, out_notes)
    loss_timing = loss_fn(label_timing, out_timing)
    loss_total = (loss_notes + loss_timing) / 2
    gradient = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradient, model.trainable_variables))
  
  return loss_total


def train(training_data, valid_data, config) -> MusicRNN:
  """
  trains the RNN

  Args:
      training_data tf.Data.dataset:
      valid_data tf.Data.dataset:
      config dictionary: 

  Returns:
      MusicRNN: fully trained RNN
  """

  # for some reason CPU is faster to train on
  # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

  config['log_dir'] = os.path.join('results', config['run_name'], 'logs')
  os.makedirs(config['log_dir'], exist_ok=True)

  model, loss_fn, optimizer = compile_model(config)
  
  # Directory where the checkpoints will be saved
  checkpoint_dir = f'./results/{config.run_name}/training_checkpoints/'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt") # _{epoch}
  checkpoint = tf.train.Checkpoint(model=model) # optimizer='RMSprop',

  config['log_dir'] = os.path.join('results', config['run_name'], 'logs', 'gradient_tape')
  os.makedirs(config['log_dir'], exist_ok=True)
  
  current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
  train_log_dir = os.path.join(config['log_dir'], current_time, 'train')
  test_log_dir = os.path.join(config['log_dir'], current_time, 'test')

  os.makedirs(train_log_dir)
  os.makedirs(test_log_dir)
  train_summary_writer = tf.summary.create_file_writer(train_log_dir)
  test_summary_writer = tf.summary.create_file_writer(test_log_dir)


  mean = tf.metrics.Mean()

  # training loop
  for epoch in range(config.epochs):
      start = time.time()

      mean.reset_states()
      for (batch_n, (inp, target)) in enumerate(training_data):
        
        loss = train_step(model=model,
          inputs=inp, 
          label_notes=target[:,:,0],
          label_timing=target[:,:,2],
          loss_fn= loss_fn,
          optimizer=optimizer)

        mean.update_state(loss)


        if batch_n % 40 == 0:
          template = f"Epoch {epoch+1} Batch {batch_n} Loss {loss:.4f}"
          print(template)

      # saving (checkpoint) the model every 5 epochs
      if (epoch + 1) % 5 == 0:
          checkpoint.save(checkpoint_prefix)

      training_accuracy = evaluate(model, data=training_data, eval_mode="training")
      testing_accuracy = evaluate(model, data=valid_data, eval_mode="validation")
      print(f'Epoch {epoch+1} Loss: {mean.result().numpy():.4f}')
      print(f'Time taken for 1 epoch {time.time() - start:.2f} sec')
      print("_"*80)

      with train_summary_writer.as_default():
        tf.summary.scalar('loss', mean.result(), step=epoch)
        tf.summary.scalar('accuracy', training_accuracy, step=epoch)
        
      with test_summary_writer.as_default():
        tf.summary.scalar('accuracy', testing_accuracy, step=epoch)

  checkpoint.save(f"{checkpoint_prefix}")

  return model



def load_model(config, training_data)->MusicRNN:



  # make model
  model = MusicRNN(
      vocab_size=config.vocab_size,
      embedding_dim=config.embedding_dim,
      rnn_units=config.rnn_units)
      # callbacks=[tensorboard_callback])

  model.compile(optimizer='Adam', loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True))

  # flush inputs through to make the graph
  for input_example_batch, _ in training_data.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions[0].shape, "# (batch_size, sequence_length, vocab_size)")

  # load weights
  # model.load_weights(checkpoint_dir)
  checkpoint = tf.train.Checkpoint(model=model)
  status = checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
  # status.assert_consumed()

  return model



def evaluate(model, data, eval_mode:str="validation"):
  """ do a step in the validation set """  
  acc = 0
  count = 0
   # validate
  for (inp, target) in data.take(3):
    logs = model(inputs=inp, states=None, return_state=False, training=False)
    preds = tf.math.argmax(logs[0], axis=1)
    acc += tf.math.count_nonzero(tf.cast(target[:,0,0], tf.int64)  == preds) / preds.shape[0]
    count+=1
  acc /= count
  print(f"{eval_mode} acc is {acc.numpy():.4f}")
  return acc