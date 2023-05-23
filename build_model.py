import tensorflow_hub as hub
import os
import tensorflow as tf
from bert import tokenization
import argparse

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def processor_model(model):
  @tf.function(input_signature=[{
        "input_word_ids": tf.TensorSpec((None, 100), tf.int32, name="input_word_ids"),
        "input_mask": tf.TensorSpec((None, 100), tf.int32, name="input_mask"),
        "segment_ids": tf.TensorSpec((None, 100), tf.int32, name="segment_ids"),
  }])
  def emotional(inputs):
     prediction = model(inputs)
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