# # Backbone Network
# We have chosen pre-trained BERT(Bidirectional Encoder Representations from Transformers) as our backbone
# network that will be transfer trained. Out of the available variations(Pre-Trained BERT Models) of BERT we
# have chosen the ”bert-base-cased” version.
# 
# BERT provides two variations(cased and uncased) for all its models. The cased version is pre-trained on
# lower case English language, while for the other one, no case transformation is done during pre-training. For
# the same reason, we will not be employing lower-casing of our input text while training in these experiments.
# Other variant details include: 12 hidden layers, a hidden size of 768, and 12 attention heads, with 110M
# parameters.
# 
# In addition to the backbone layer we add the input and output layers. The size of the input layer is 3(one
# each for: Input word ID, Input Mask and Segment ID) We flatten the output received from the BERT layers
# and feed it to our classification layer with output size 14(matching our emotion set) and sigmoid activation.
# For tokenizing our inputs, we use the WordPiece Tokenizer implementation provided by BERT. WordPiece
# encodes text into subword tokens that can be fed into the BERT model. WordPiece is a type of subword tok-
# enization that can handle out-of-vocabulary(OOV) words by breaking them down into smaller subword units.
# The resulting subword tokens are then mapped to a fixed-size vocabulary of subword units. During training
# and inference, the BERT model uses these subword embeddings to represent the input text. By using subword
# tokenization, the BERT tokenizer is able to handle OOV words and capture meaningful subword units that can
# help the model better understand the structure and meaning of the text. 
# 
# Handling OOV words is essential for
# our model as we are processing data retrieved from an online message posting board.
# 
# ### The helper functions that create and intialize the network are below

#Pretrained BERT
import tensorflow_hub as hub

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)

from bert import tokenization
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import Adagrad
from tensorflow.keras.optimizers.experimental import SGD
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint

import numpy as np

# The 4 next lines allows to prevent an error due to Bert version
import sys
from absl import flags
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

def build_model(bert_layer, train_labels, add_dropout=False, max_len=512, loss_fn="categorical_crossentropy", 
                full_train=True, add_dense=False, optimizer="Adam", activation='sigmoid'):
    input_word_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = layers.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = layers.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    sequence_output = None
    
    if full_train:
        _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    else:
        _, sequence_output = bert_layer_frozen([input_word_ids, input_mask, segment_ids])
    
    clf_output = sequence_output[:, 0, :]

    flatten = layers.Flatten(name='flatten')
    output_flatten = flatten(clf_output)

    if add_dropout:
        output_flatten = tf.keras.layers.Dropout(0.1, name="dropout")(output_flatten)
    #l = tf.keras.layers.Dense(len(np.unique(train_labels)), activation='sigmoid', name="output")(l)
    
    if add_dense:
        output_flatten = layers.Dense(768, activation='relu')(output_flatten)

    out = layers.Dense(len(np.unique(train_labels)), activation=activation)(output_flatten)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    METRICS = [
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),      
    ]
    if optimizer == "Adam":
        print("Selected Adam as Optimizer")
        model.compile(Adam(lr=2e-6), loss=loss_fn, metrics=METRICS)
    elif optimizer == "Adagrad":
        model.compile(Adagrad(lr=2e-6), loss=loss_fn, metrics=METRICS)
    elif optimizer == "SDG":
        model.compile(SDG(lr=2e-6), loss=loss_fn, metrics=METRICS)
        
    return model