import warnings
warnings.filterwarnings('ignore')

#import absl
#import absl.logging
#absl.logging.set_verbosity(absl.logging.ERROR)

#Data processing
import numpy as np 
import pandas as pd

#Plotting
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.offline as pyo
import plotly.graph_objs as go

#Setup plots
import plotly.io as pio
sns.set_style("whitegrid")
sns.set_context(rc = {'patch.linewidth': 0.0})

#Data Handling
import json
import glob
import os
import math
import pandas
import tensorflow as tf

from utils import replace_orig_emo_with_new_emotions, generate_new_emotions_column

from model import bert_encode
from model_train import train_and_run_model

#flags
save_model = True
save_figs = True
save_embeddings = False

#pull the data from disk for analysis, training and test
df_train = pd.read_csv("data/train.tsv", sep='\t', header=None, names=['text', 'emotion', 'ID'])
df_dev = pd.read_csv("data/dev.tsv", sep='\t', header=None, names=['text', 'emotion', 'ID'])
df_test = pd.read_csv("data/test.tsv", sep='\t', header=None, names=['text', 'emotion', 'ID'])

# # Data set Preparation and Split
# 
# To prepare the data set for building a multi-class classifier based on new emotion groupings, we essentially need
# to make the following two changes to the data set:
# 1. Removing the multi-label attributes for each example in the data set.
# To do this, for data entries having more than one class label, we only considered the first class label and
# ignored the others.
# 2. Replacing the old emotion labels with new ones. This is picked from the mapping files from the data folder.
# The data set already comes pre-split into 3 different files(train.tsv, dev.tsv,test.tsv). For our experiments,
# we have chosen to utilise the train.tsv and dev.tsv files for training and validating the data set. The test.tsv
# file was used for testing our experiments.

print("Preparing data...")
df_analysis = pd.concat([df_train, df_dev], sort=False)
replace_orig_emo_with_new_emotions(df_analysis, "emotion", "newemotion", "data/map_numeric.txt")
df_analysis = df_analysis.drop(columns=["emotion"])
df_analysis.rename(columns={"newemotion": "emotion"}, inplace=True)  

# # Test Setup
# 
# Throughout our experimentation, we will be testing our experimental setup on the test data
# set and monitor precision, recall and F1 scores per class to observe the models’s performance.
# Additionally, the Confusion matrix, PR and ROC curves will be plotted wherever deemed nec-
# essary. TSNE will also be used to generate embeddings.
# 
# ## The helper functions and test data setup is in the couple of cells below

#prepare for test
import tensorflow_hub as hub
from bert import tokenization

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

print("Preparing test...")
df_test = pd.read_csv("data/test.tsv", sep='\t', header=None, names=['text', 'emotion', 'ID'])
df_test['emotion'] = df_test['emotion'].str.split(',').str[0]
replace_orig_emo_with_new_emotions(df_test, "emotion", "newemotion", "data/map_numeric.txt")
y_test = df_test["newemotion"].to_numpy().astype(int)
x_test = bert_encode(df_test.text.values, tokenizer, max_len=100)

#prepare DF for multi-class
print("Preparing multi class analysis...")
df_multi_class_analysis = pd.concat([df_train, df_dev], sort=False)
df_multi_class_analysis['emotion'] = df_multi_class_analysis['emotion'].str.split(',').str[0]
df_multi_class_analysis['emotionid'] = df_multi_class_analysis['emotion']
df_multi_class_analysis = df_multi_class_analysis.reset_index()
replace_orig_emo_with_new_emotions(df_multi_class_analysis, "emotionid", "new_emotion_id", "data/map_numeric.txt")
df_multi_class_analysis = df_multi_class_analysis.reset_index()
generate_new_emotions_column(df_multi_class_analysis)
df_multi_class_analysis = df_multi_class_analysis.drop(columns={"emotion"})
df_multi_class_analysis = df_multi_class_analysis.rename({'newemotion': 'emotion'}, axis=1)

#under-sampling
reduce_emotions = set()
emotion_target = 4000
emotion_counts_orig = df_multi_class_analysis['emotion'].value_counts()

#over sampling
def get_replication_vals(row):
    emotion_count = emotion_counts_orig[row['emotion']]
    if(emotion_count < emotion_target):
        return int(emotion_target/emotion_count)
    else:
        reduce_emotions.add(row['emotion'])
        return 1

print("Balancing data...")
df_analysis_balanced_data = df_multi_class_analysis.copy()
df_analysis_balanced_data['rep_count'] = df_analysis_balanced_data.apply(get_replication_vals, axis=1);
df_analysis_balanced_data = df_analysis_balanced_data.loc[df_analysis_balanced_data.index.repeat(df_analysis_balanced_data['rep_count'])].assign(fifo_qty=1).reset_index(drop=True);
df_analysis_balanced_data = df_analysis_balanced_data.reindex()
df_analysis_balanced_data = df_analysis_balanced_data.drop(columns=['rep_count', 'fifo_qty'])
print("\n*******New number of examples:", df_analysis_balanced_data['emotion'].value_counts().sum())

early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10)

print("Starting training...")
model_batch_4, inputs_batch_4 = train_and_run_model(df_analysis_balanced_data, 
                                                                "emotion", "new_emotion_id", 
                                                                "model_batch_4",True,batch_size=4, 
                                                                loss_fn=tf.keras.losses.KLDivergence(),activation=tf.keras.activations.softmax)
test_and_generate_curves(model_batch_4, "model_batch_4")
display_embeddings(model_batch_4, inputs_batch_4, df_analysis_balanced_data,"model_batch_4")