import sklearn
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

    
new_emotion_strings = None
with open('data/new_emotions.txt') as f:
    new_emotion_strings = f.read().splitlines()

g_without_neutral = False

def display_embeddings(model,train_input,df_analysis, model_name):
    
    print("Extracting Embeddings...")
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer('flatten').output)
    sentence_embedded = intermediate_layer_model.predict(train_input)

    X = np.array(sentence_embedded)
    X_embedded = TSNE(n_components=2).fit_transform(X)

    print("Plotting Embeddings...")
    df_embeddings = pd.DataFrame(X_embedded)
    df_embeddings = df_embeddings.rename(columns={0:'x',1:'y'})
    df_embeddings = df_embeddings.assign(label=df_analysis.emotion.values)
    df_embeddings = df_embeddings.assign(text=df_analysis.text.values)
    
    fig = px.scatter(
    df_embeddings, x='x', y='y',
    color='label', labels={'color': 'label'},
    hover_data=['text'], title = 'GoEmotions Embedding Visualization',
    color_discrete_sequence=px.colors.qualitative.Light24)

    fig.show(plotly_renderer)
    if save_embeddings:
        fig.write_image(f'models/{model_name}/embeddings.jpeg')
    
def gen_pred_list(label):
    one_hot_label = np.zeros(shape=(1, 13 if g_without_neutral else 14))
    one_hot_label[0, int(label)] = 1
    return one_hot_label

def create_confusion_matrix(y_test, y_pred, model, model_name, without_neutral=False):
    #Predict
    matrix = confusion_matrix(y_test, y_pred.argmax(axis=1))
    
    emotions = new_emotion_strings.pop() if without_neutral else new_emotion_strings
    df_cm = pd.DataFrame(matrix, columns=emotions, index = emotions)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize = (15,15))
    sns.set(font_scale=1.4)#for label size
    sns.heatmap(df_cm, cmap=sns.cubehelix_palette(as_cmap=True), annot=True,annot_kws={"size": 6}, fmt='g')# font size
    plt.title(f'Test - Confusion Matrix for emotion classification for model: {model_name}')
    if save_figs:
        os.makedirs(f'models/{model_name}/', exist_ok=True)
        plt.savefig(f'models/{model_name}/Confusion_matrix.jpeg')
    
def plot_roc_curve(y_test, y_score, model, model_name, without_neutral=False):
    y_test_le = np.vectorize(gen_pred_list, signature='()->(n)')(y_test)
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(13 if without_neutral else 14):
        fpr[i], tpr[i], _ = sklearn.metrics.roc_curve(y_test_le[:, i], y_score[:, i])
        roc_auc[i] = sklearn.metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.roc_curve(y_test_le.ravel(), y_score.ravel())
    roc_auc["micro"] = sklearn.metrics.auc(fpr["micro"], tpr["micro"])

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]))
    
    emotions = new_emotion_strings.pop() if without_neutral else new_emotion_strings
        
    for i in range(len(emotions)):
        plt.plot(fpr[i], tpr[i], label='ROC curve of {0} (area = {1:0.2f})'
                                   ''.format(emotions[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Test - ROC Curve for multi-class emotion classification for model: {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    plt.draw()
    if save_figs:
        os.makedirs(f'models/{model_name}/', exist_ok=True)
        plt.savefig(f'models/{model_name}/ROC_Curve.jpeg', dpi=100)

def plot_pr_curve(y_test, y_score, model, model_name, without_neutral=False):
    y_test_le = np.vectorize(gen_pred_list, signature='()->(n)')(y_test)
    
    fpr = dict()
    tpr = dict()
    for i in range(13 if without_neutral else 14):
        fpr[i], tpr[i], _ = sklearn.metrics.precision_recall_curve(y_test_le[:, i], y_score[:, i])

    # Compute micro-average PR curve
    fpr["micro"], tpr["micro"], _ = sklearn.metrics.precision_recall_curve(y_test_le.ravel(), y_score.ravel())

    # Plot PR curve
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average PR curve')
        
    emotions = new_emotion_strings.pop() if without_neutral else new_emotion_strings
    #print(emotions)
    for i in range(len(emotions)):
        plt.plot(fpr[i], tpr[i], label='PR curve of class {0}'.format(emotions[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Test - PR Curve for multi-class emotion classification for model: {model_name}')
    plt.legend(loc="lower right")
    plt.show()
    plt.draw()
    
    if save_figs:
        os.makedirs(f'models/{model_name}/', exist_ok=True)
        plt.savefig(f'models/{model_name}/PR_Curve.jpeg', dpi=100)


def test_and_generate_curves(model, model_name, without_neutral=False):
    #test and predict
    y_pred = model.predict(x_test)
    
    global g_without_neutral
    g_without_neutral = without_neutral
    
    y_test_analysis = []
    if without_neutral:
        for sublist in y_test:
            sublist = np.delete(sublist, -1)
            y_test_analysis.append(sublist)
    
    y_test_analysis = y_test
    #y_test_analysis = np.array([np.array(xi) for xi in y_test_analysis])
    #y_test_analysis = np.array(y_test_analysis)

    y_true = np.vectorize(gen_pred_list, signature='()->(n)')(y_test_analysis)
    y_pred_support = (y_pred > 0.5) 

    #MAE and MSE
    from sklearn.metrics import precision_recall_fscore_support, accuracy_score
    results = dict()

    results["macro_precision"], results["macro_recall"], results["macro_f1"], _ = precision_recall_fscore_support(y_true, y_pred_support, average="macro")
    results["micro_precision"], results["micro_recall"], results["micro_f1"], _ = precision_recall_fscore_support(y_true, y_pred_support, average="micro")
    results["weighted_precision"], results["weighted_recall"], results["weighted_f1"], _ = precision_recall_fscore_support(y_true, y_pred_support, average="weighted")
    results["precision"], results["recall"], results["f1"], _ = precision_recall_fscore_support(y_true, y_pred_support, average=None)
    
    emotions = new_emotion_strings.pop() if without_neutral else new_emotion_strings

    result_list = []
    for idx, label in enumerate(emotions):
        result_list.append([label, results['precision'][idx], results['recall'][idx], results['f1'][idx]])
    df_result = pd.DataFrame(result_list)
    df_result = df_result.sort_values(by=[0])
    df_result = df_result.append({0:'weighted-average', 1:results['weighted_precision'], 2:results['weighted_recall'], 3:results['weighted_f1']}, ignore_index=True)
    df_result.columns = ['Emotion', 'Precision', 'Recall', 'F1']
    df_result

    print("Mean Absolute Error:", sklearn.metrics.mean_absolute_error(y_true, y_pred))
    print("Mean Squared Error:", sklearn.metrics.mean_squared_error(y_true, y_pred))
    print(df_result)
    
    if save_model:
        os.makedirs(f'models/{model_name}/', exist_ok=True)
        df_result.to_csv(f"models/{model_name}/metrics.csv")
    
    #Confusion Matrix
    create_confusion_matrix(y_test_analysis,y_pred,model,model_name,without_neutral)
    # ROC curve
    plot_roc_curve(y_test_analysis,y_pred,model,model_name,without_neutral)
    # PR Curve
    plot_pr_curve(y_test_analysis,y_pred,model,model_name,without_neutral)


def replace_orig_emo_with_new_emotions(df_analysis, column, new_column_name, mapping_file):
    #new_emotions
    new_emotion_nums = None
    with open(mapping_file) as f:
        new_emotion_nums = f.read().splitlines()
    df_analysis["newemotion"] = "";
    for index, row in df_analysis.iterrows():
        orig_emotions_list = row[column].split(",")
        for emo in orig_emotions_list:
            new_emotions_str = ""
            #print(emo)
            if(emo != ''):
                #print(emo)
                if(new_emotions_str != ''):
                    new_emotions_str = new_emotions_str + "," + new_emotion_nums[int(emo)]
                else:
                    new_emotions_str = new_emotion_nums[int(emo)]
                
            #print("old: ", emo, " new: ", new_emotions_str)
        df_analysis.at[index, new_column_name] = new_emotions_str
        
def generate_new_emotions_column(conversion_df):

    conversion_df['newemotion'] = conversion_df['emotion'] 

    emotion_strings = None
    with open('data/emotions.txt') as f:
        emotion_strings = f.read().splitlines()
        
    new_emotion_strings = None
    with open('data/map.txt') as f:
        new_emotion_strings = f.read().splitlines()
    
    #original_emotions
    conversion_df["emotion"] = pd.to_numeric(conversion_df["emotion"])
    conversion_df.loc[0:,'emotion'] = conversion_df.loc[0:,'emotion'].map(dict(enumerate(emotion_strings)))

    
    #new_emotions
    conversion_df["newemotion"] = pd.to_numeric(conversion_df["newemotion"])
    conversion_df.loc[0:,'newemotion'] = conversion_df.loc[0:,'newemotion'].map(dict(enumerate(new_emotion_strings)))