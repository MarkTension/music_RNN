
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

  for temperature in [1, 2, 4]:
    one_step_model = OneStep(model, temperature=temperature)
    state = None
    samples = []
    attentions = [] 
    probabilities = []
    entropies = []
    
    training_data = training_data.shuffle(32)
    dataset =  training_data.take(1)
    input = tf.constant(list(dataset.as_numpy_iterator())[0][0][:1])#[:,:10,:]

    # sample loop
    for _n in range(config.num_notes_sampled * 4):
      pred, state, attention, logits, pred_entropy = one_step_model.generate_one_step(
          input, states=state)
      
      timing = tf.constant(np.floor((_n % 16)/4), dtype=tf.int64, shape=[1,1,1])
      prediction = tf.reshape(pred, shape=[1,1,1])
      prediction = tf.concat([prediction, timing], axis=2)
      input = tf.concat([input, prediction], axis=1)
      input = input[:,1:,:]
      
      samples.append(pred.numpy())
      attentions.append(attention[:,:,0].numpy())
      probabilities.append([logits[0,:]]) #.numpy())
      entropies.append([pred_entropy])

    # reconstruct og format
    samples = output_raw_to_voices(samples)
    attentions = np.mean(output_raw_to_voices(np.array(attentions)), axis=0)
    probabilities = np.max(np.mean(output_raw_to_voices(probabilities), axis=0), axis=0)
    entropies_raw = output_raw_to_voices(entropies)
    entropies = np.mean(output_raw_to_voices(entropies), axis=0)

    # save to json
    df = pd.DataFrame({"note0": samples[0], "note1": samples[1], "note2": samples[2], "note3": samples[3], 
    "attention": attentions.tolist(), "probs": probabilities.tolist(), "entropies": entropies.tolist()
    })

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
    predicted_logits, states, context = self.model(inputs=inputs, states=states,
                                          return_state=True, training=False)
    # Only use the last prediction.
    # predicted_logits = predicted_logits[:, -1, :]
    predicted_logits = predicted_logits/self.temperature

    # Sample the output logits to generate token IDs.
    predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
    predicted_ids = tf.squeeze(predicted_ids, axis=-1)
    predicted_logits = tf.keras.activations.softmax(predicted_logits).numpy()

    pred_entropy = entropy(predicted_logits[0,:])
    # plt.bar(range(len(predicted_logits[0])), predicted_logits[0])
    # plt.title(f"Note probability distribution with temperature {self.temperature}")
    # plt.show()
    # plt.cla()
    

    # do softmax
    # Return the characters and model state.
    return predicted_ids, states, context, predicted_logits, pred_entropy