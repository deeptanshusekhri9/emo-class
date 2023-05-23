import tensorflow_hub as hub
import os
import tensorflow as tf
from bert import tokenization
import pandas as pd
import argparse
from model import bert_encode, build_model

def processor(model):
  
  @tf.funtion(input_signature=[tf.TensorSpec([None], tf.string)])
  def emotional(text):
     module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
     bert_layer = hub.KerasLayer(module_url, trainable=True)
     bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)

     vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
     do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
     tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
     tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
     predict_input = bert_encode(text, tokenizer, max_len=100)
     output = model(predict_input)
     labels=['affection','anticipation','anxious','confusion','curiosity','delight','desire',
             'frustration','melancholy','mortification','relief','respect','revelation','neutral']
     predict_output = labels[output.argmax(axis=-1)]
  
     return predict_output

def main(args):
  
  print(f'loading {args.load_model_paths}')
  model = tf.keras.models.load_model(args.load_model_paths)
  tf.saved_model.save(
                      model,export_dir=os.path.join(args.save_model_dir,str(args.version)),
                      signatures={"serving_default": processor(model)},
                      )


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # Input Arguments
  parser.add_argument(
      '--load_model_paths',
      help='path to model to load',
      required=True
  )
  parser.add_argument(
      '--save_model_dir',
      help="""\
      Directory to save model in.\
      """,
      required=True
)
  parser.add_argument(
      '--version',
      help="""\
      Version of the model.\
      """,
      type=int,
      required=True
)
  args = parser.parse_args()
  arguments = args.__dict__
  main(args)