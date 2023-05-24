import tensorflow_hub as hub
from bert import tokenization

import requests
import json
from model import bert_encode
import grpc
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

sentence = "can I get a prediction for this good code."

# Tokenize the sentence but this time with TensorFlow tensors as output already batch sized to 1. Ex:
# {
#    'input_ids': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[  101, 19082,   102]])>,
#    'token_type_ids': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[0, 0, 0]])>,
#    'attention_mask': <tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[1, 1, 1]])>
# }
#batch = tokenizer(sentence, return_tensors="tf")
batch = bert_encode(sentence, tokenizer, max_len=100)

# Create a channel that will be connected to the gRPC port of the container
channel = grpc.insecure_channel("0.0.0.0:8500/models/1:predict")

# Create a stub made for prediction. This stub will be used to send the gRPC request to the TF Server.
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

# Create a gRPC request made for prediction
request = predict_pb2.PredictRequest()

# Set the name of the model, for this use case it is bert
request.model_spec.name = "1"

# Set which signature is used to format the gRPC query, here the default one
request.model_spec.signature_name = "1"

# Set the input_ids input from the input_ids given by the tokenizer
# tf.make_tensor_proto turns a TensorFlow tensor into a Protobuf tensor
request.inputs["input_word_ids"].CopyFrom(tf.make_tensor_proto(batch[0]))

# Same with attention mask
request.inputs["input_mask"].CopyFrom(tf.make_tensor_proto(batch[1]))

# Same with token type ids
request.inputs["segment_ids"].CopyFrom(tf.make_tensor_proto(batch[2]))

# Send the gRPC request to the TF Server
result = stub.Predict(request)

# The output is a protobuf where the only one output is a list of probabilities
# assigned to the key logits. As the probabilities as in float, the list is
# converted into a numpy array of floats with .float_val
#output = result.outputs["logits"].float_val

# Print the proper LABEL with its index
#print(config.id2label[np.argmax(np.abs(output))])
print(result)