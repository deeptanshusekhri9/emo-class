import tensorflow_hub as hub
from bert import tokenization

import requests
import json
from model import bert_encode

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

sentence = "can I get a prediction for this good code."

# Tokenize the sentence
#batch = tokenizer(sentence)
batch = bert_encode(sentence, tokenizer, max_len=100)

# Convert the batch into a proper dict
batch = dict(batch)

# Put the example into a list of size 1, that corresponds to the batch size
batch = [batch]

# The REST API needs a JSON that contains the key instances to declare the examples to process
input_data = {"instances": batch}

# Query the REST API, the path corresponds to http://host:port/model_version/models_root_folder/model_name:method
r = requests.post("http://localhost:8501/v1/models/bert:predict", data=json.dumps(input_data))

# Parse the JSON result. The results are contained in a list with a root key called "predictions"
# and as there is only one example, takes the first element of the list
result = json.loads(r.text)["predictions"][0]

print(result)