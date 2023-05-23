import tensorflow_hub as hub
import os
import tensorflow as tf
from bert import tokenization
import argparse
from model import bert_encode

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def processor_model(model):
  @tf.function(input_signature=[tf.TensorSpec(shape=(), dtype=tf.string)])
  def emotional(text):
     global tokenizer
     print(type(text))
     #text = tf.strings.(text)
     #text = text.decode("utf-8")
     #TODO: fix encoder input text
     predict_input = bert_encode("my test", tokenizer, max_len=100)
     prediction = model(predict_input)
     #prediction = model(text)
     labels=['affection','anticipation','anxious','confusion','curiosity','delight','desire',
             'frustration','melancholy','mortification','relief','respect','revelation','neutral']
     indices = tf.argmax(prediction, axis=-1)  # Index with highest prediction
     label = tf.gather(params=labels, indices=indices)  # Class name
     return label
  
  return emotional

def main(args):
  
  print(f'loading {args.load_model_paths}')
  model = tf.keras.models.load_model(args.load_model_paths)
  tf.saved_model.save(
                      model,export_dir=os.path.join(args.save_model_dir,str(args.version)),
                      signatures={"serving_default": processor_model(model)},
                      )


if __name__ == '__main__':
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  # Input Arguments
  parser.add_argument(
      "--load_model_paths",
      help='path to model to load',
      type=str,
      default="/Users/nehasachdeva/Documents/MSc-AI_Docs/NLP/coursework/deeptanshu_NLP_CW/emo-class/models/golden/",
  )
  parser.add_argument(
      '--save_model_dir',
      help="""\
      Directory to save model in.\
      """,
      type=str,
      default="/Users/nehasachdeva/Documents/MSc-AI_Docs/NLP/coursework/deeptanshu_NLP_CW/emo-class/models/deploy/"
  )
  parser.add_argument(
      '--version',
      help="""\
      Version of the model.\
      """,
      type=int,
      default="1",
  )
  args = parser.parse_args()
  arguments = args.__dict__
  main(args)