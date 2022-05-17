
from datetime import datetime
from midi_utils import get_midi_from_numbers
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy.stats import entropy
from matplotlib import pyplot as plt

def sample(model, config, training_data):

  timestamp = datetime.now().strftime("%m_%d_%H_%M")

  for temperature in [1, 2, 4]:
    one_step_model = OneStep(model, temperature=temperature)
    state = None
    result = []
    attention_out = []
    logits_out = []
    entropy_out = []

    # input = tf.constant([[[12, 5]]], dtype=tf.int64)
    dataset =  training_data.take(1)
    input = tf.constant(list(dataset.as_numpy_iterator())[0][0][:1])

    # sample loop
    for _n in range(config.num_notes_sampled):
      pred, state, alpha, logits, pred_entropy = one_step_model.generate_one_step(
          input, states=state)
      
      timing = tf.constant(np.floor((_n % 16)/4), dtype=tf.int64, shape=[1,1,1])
      prediction = tf.reshape(pred, shape=[1,1,1])
      prediction = tf.concat([prediction, timing], axis=2)
      input = tf.concat([input, prediction], axis=1)
      input = input[:,1:,:]
      result.append(pred)
      attention_out.append(alpha[:,:,0].numpy())
      logits_out.append(logits[0,:]) #.numpy())
      entropy_out.append(pred_entropy)
    
    # reconstruct og format
    new = [[],[],[],[]]
    for i, el in enumerate(result):
        new[i % 4].extend(el.numpy())

    result = np.array(new)

    attention = [[],[],[],[]]
    for i, el in enumerate(attention_out):
        attention[i % 4].append(el[0])

    logits = [[],[],[],[]]
    for i, el in enumerate(logits_out):
        logits[i % 4].append(el)
    
    entropies = [[],[],[],[]]
    for i, el in enumerate(entropy_out):
        entropies[i % 4].append(el)


    # save to json
    df = pd.DataFrame({"note0": result[0], "note1": result[1], "note2": result[2], "note3": result[3], 
    "attention0": attention[0], "attention1": attention[1], "attention2": attention[2], "attention3": attention[3],
    "probs0": logits[0], "probs1": logits[1], "probs2": logits[2], "probs3": logits[3],
    "entropy0": entropies[0], "entropy1": entropies[1], "entropy2": entropies[2], "entropy3": entropies[3],
    })
    json_out_dir = os.path.join("results", config.run_name, f"json_{timestamp}_epochs_{config.epochs}_temp_{temperature}.json")

    df.to_json(json_out_dir)

    midi = get_midi_from_numbers(result.transpose(), config.data_details['version'])
    midi_out_dir = os.path.join("results", config.run_name, f"midi_{timestamp}_epochs_{config.epochs}_temp_{temperature}.mid")
    # midi.save("test.mid")
    midi.save(midi_out_dir)



class OneStep(tf.keras.Model):
  def __init__(self, model, temperature=1.0):
    super().__init__()
    self.temperature = temperature
    self.model = model

  @tf.function
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