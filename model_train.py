import tensorflow_hub as hub
import tensorflow as tf
from bert import tokenization
import pandas as pd

from model import bert_encode, build_model

module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1"
bert_layer = hub.KerasLayer(module_url, trainable=True)
bert_layer_frozen = hub.KerasLayer(module_url, trainable=False)

vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

#we are putting an early stopping criteria so that the training stops of the validation loss is not improving for 2 epochs
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2)

def convert_to_one_hot_label(label, training_label_len):
    one_hot_label = pd.DataFrame(0, index=range(1), columns=range(training_label_len))
    labels = label.split(",")
    for l in labels:
        one_hot_label[int(l)] = 1
    return one_hot_label

def train_and_run_model(df_analysis, named_labels_column, labels_column, model_name, 
                        add_dropout=False, validation_split=0.2, epochs=10, batch_size=32,
                       loss_fn="categorical_crossentropy",full_train=True, add_dense=False, optimizer="Adam",
                       activation='sigmoid'):
    train_input = bert_encode(df_analysis.text.values, tokenizer, max_len=100)

    train_labels = df_analysis[named_labels_column].str.split(',').explode('Cast').value_counts().index.tolist()
    training_label_len = len(train_labels)
    print("No of Training Labels:" , len(train_labels))

    model = build_model(bert_layer, train_labels, add_dropout, max_len=100,loss_fn=loss_fn,
                        full_train=full_train,add_dense=add_dense, optimizer=optimizer,
                       activation=activation)
    df_one_hot = pd.concat([convert_to_one_hot_label(a, training_label_len) for a in df_analysis[labels_column]])
    
    print("Training Model with: Epochs:{} Split:{} Batch Size:{}".format(epochs, validation_split, batch_size))
          
    train_history = model.fit(
        train_input, df_one_hot,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping_callback]
    )
    
    if save_model:
        print("Saving Model...")
        model.save("models/{}".format(model_name))
        print("Model saved at: models/{}".format(model_name))
    
    return model, train_input