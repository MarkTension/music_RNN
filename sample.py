
from datetime import datetime
from midi_utils import get_midi_from_numbers
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.stats import entropy
from matplotlib import pyplot as plt


def output_raw_to_voices(raw_list):
    output = [[],[],[],[]]
    for i, item in enumerate(raw_list):
        output[i % 4].extend(item)
    return np.array(output)


def sample(model, config, training_data):

  timestamp = datetime.now().strftime("%m_%d_%H_%M")

  for temperature in [1, 1.5]:
    one_step_model = OneStep(model, temperature=temperature)
    state = None
    samples = []
    attentions = [] 
    probabilities = []
    entropies = []
    means = []
    stds = []
    
    training_data = training_data.shuffle(32)
    dataset =  training_data.take(1)
    input = tf.constant(list(dataset.as_numpy_iterator())[0][0][:1])#[:,:10,:]

    # sample loop
    for _n in range(config.num_notes_sampled * 4):
      pred_notes, pred_onoff, state, attention, logits, pred_entropy, mean, std = one_step_model.generate_one_step(
          input, states=state) # TODO: think about if I want to return state here
      
      timing = tf.constant(np.floor((_n % 16)/4), dtype=tf.int64, shape=[1,1,1])
      prediction = tf.reshape(pred_notes, shape=[1,1,1])
      onoff = tf.reshape(pred_onoff, shape=[1,1,1])
      prediction = tf.concat([prediction, timing, onoff], axis=2)
      # [1, 128, 3] # 
      input = tf.concat([tf.cast(input, tf.float32), tf.cast(prediction, tf.float32)], axis=1)
      # advance one step
      input = input[:,1:,:]
      
      samples.append(pred_notes.numpy())
      attentions.append(attention[:,:,0].numpy())
      probabilities.append([logits[0,:]]) #.numpy())
      entropies.append([pred_entropy])
      means.append([mean])
      stds.append([std])

    # reconstruct og format
    samples = output_raw_to_voices(samples)
    probabilities = np.max(np.mean(output_raw_to_voices(probabilities), axis=0), axis=1)
    entropies_raw = output_raw_to_voices(entropies)
    attentions = np.mean(output_raw_to_voices(np.array(attentions)), axis=0)
    entropies = np.mean(output_raw_to_voices(entropies), axis=0)
    means = np.mean(output_raw_to_voices(means), axis=0)
    stds = np.mean(output_raw_to_voices(stds), axis=0)

    # save to json
    df = pd.DataFrame({"note0": samples[0], "note1": samples[1], "note2": samples[2], "note3": samples[3], 
    "attention": attentions.tolist(), "probs": probabilities.tolist(), "entropies": entropies.tolist(),
    "means": means.tolist(), "stds": stds.tolist()
    })

    for col in df.columns:
      if 'note' in col:
        df[col] = df.loc[df[col] == config['data_details']['token_empty_note'], col] = 0
        df[col] = df.loc[df[col] == config['data_details']['token_reset_song'], col] = 0

    df['means'] = df['means'] / np.max(np.abs(df['means']))
    df['stds'] = df['stds'] / np.max(np.abs(df['stds']))


    json_out_dir = os.path.join("results", config.run_name, f"json_{timestamp}_epochs_{config.epochs}_temp_{temperature}.json")

    df.to_json(json_out_dir)

    midi = get_midi_from_numbers(samples.transpose(), config.data_details['version'], velocities=entropies_raw)
    midi_out_dir = os.path.join("results", config.run_name, f"midi_{timestamp}_epochs_{config.epochs}_temp_{temperature}.mid")
    # midi.save("test.mid")
    midi.save(midi_out_dir)



class OneStep(tf.keras.Model):
  def __init__(self, model, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model

#   @tf.function
  def generate_one_step(self, inputs, states=None):

    # Run the model.
    # predicted_logits.shape is [batch, char, next_char_logits]
    predicted_logits, predicted_logits_onoff, states, context = self.model(inputs=inputs, states=states,
                                          return_state=True, training=False)

    # Only use the last prediction.
    predicted_logits = predicted_logits/self.temperature

    mean= tf.math.reduce_mean(predicted_logits).numpy()
    std = tf.math.reduce_std(predicted_logits).numpy()
    max = tf.math.reduce_max(predicted_logits).numpy()

    # print(f"coords are mean = {mean}, std = {std}, max={max}")
    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)
    
    # sample onoff predictions
    predicted_ids_onoff = tf.random.categorical(predicted_logits_onoff, num_samples=1)
    predicted_ids_onoff = tf.squeeze(predicted_ids_onoff, axis=-1)

    predicted_logits = tf.keras.activations.softmax(predicted_logits).numpy()
    pred_entropy = entropy(predicted_logits[0,:])
    
    return predicted_ids, predicted_ids_onoff, states, context, predicted_logits, pred_entropy, mean, std