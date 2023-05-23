import tensorflow_hub as hub

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)

from bert import tokenization
import tensorflow as tf
from tensorflow.keras.models import Model

import numpy as np
from tensorflow import keras

from model import bert_encode
import pandas as pd
from utils import replace_orig_emo_with_new_emotions

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

model = keras.models.load_model('models/golden')

model.summary()

new_emotion_strings = None
with open('data/new_emotions.txt') as f:
    new_emotion_strings = f.read().splitlines()
 
correct = 0
incorrect = 0   
print("Preparing test...")
df_test = pd.read_csv("data/test.tsv", sep='\t', header=None, names=['text', 'emotion', 'ID'])
df_test['emotion'] = df_test['emotion'].str.split(',').str[0]
replace_orig_emo_with_new_emotions(df_test, "emotion", "newemotion", "data/map_numeric.txt")

idx = 0
for index, row in df_test.iterrows():
    # Access the data in each column for the current row
    idx+=1
    text = row['text']
    emotion = row['newemotion']  
    values = (model.predict(bert_encode([text], tokenizer, max_len=100)))
    print(values.argmax())
    print(emotion)

    if(values.argmax() == int(emotion)):
        correct+=1
    else:
        incorrect+=1
    
    if idx > 100 :
        break

print("Correct predictions{}, incorrect predictions:{}".format(correct, incorrect))
#print(new_emotion_strings[values.argmax()])