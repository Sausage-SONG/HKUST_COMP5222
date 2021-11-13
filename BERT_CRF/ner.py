import numpy as np

from tensorflow.keras import *
import tensorflow as tf
from tensorflow.keras import *

from sklearn.metrics import classification_report
from crf import *

def createNEROutputs(texts,spans,max_length,tokenizer):
    outputs = []
    for text,span in zip(texts,spans):
        output = np.zeros(max_length*3,dtype=np.float).reshape((max_length,3))
        tokens = tokenizer.tokenize(text)[:max_length]
        length = 0
        start = True
        for i in range(len(tokens),max_length):
            output[i,0] = 1.0
        for index,token in enumerate(tokens):
            sub = False
            if "##" in token:
                sub = True
                token = token[2:]
            if not start:
                next_index = text[length:].find(token)
                if next_index == 0:
                    sub = True
                length += next_index
            if length in span:
                output[index,2] = 1.0
                output[index,0] = 0.0
            else:
                output[index,1] = 1.0
                output[index,0] = 0.0
            length += len(token)
            start = False
        outputs.append(output)
    return np.array(outputs)


def NERGetIndicesSingleText(outputs,text,tokenizer):
    outputs = tf.argmax(outputs,axis=-1)
    tokens = tokenizer.tokenize(text)
    index = 0
    indexes = []
    sub = False
    prev = False
    for token,output in zip(tokens,outputs):
        if token[:2] == "##":
            token = token[2:]
            sub = True
        else:
            sub = False
        temp_index = text[index:].find(token)
        temp_start = index+temp_index
        if output == 2 or (sub and prev and output != 0):
            prev = True
            indexes = indexes + list(range(temp_start,temp_start+len(token)))
        else:
            prev = False
        index = temp_start+len(token)
    return np.array(indexes)

def createIndicesForNERModel(predicts,texts,tokenizer):
    outputs = []
    for text,pred in zip(texts,predicts):
         indices = NERGetIndicesSingleText(pred,text,tokenizer)
         outputs.append(indices)
    return outputs


def createInputForNER(texts,max_length,tokenizer):
    input_length = []
    for text in texts:
        input_length.append(min(max_length,len(tokenizer.tokenize(text))))
    tokens = tokenizer(texts,padding="max_length",max_length=max_length,return_tensors="tf",truncation=True)
    data = [np.array(tokens['input_ids']),np.array(tokens['token_type_ids']),np.array(tokens['attention_mask']),np.array(input_length)]
    return data

def createToxicModelWithGivenBaseModel(max_input_length,base_model):
    input_ids_layer = layers.Input(shape=(max_input_length,),name="encoder_input_ids",dtype=tf.int32)
    input_type_ids_layer = layers.Input(shape=(max_input_length,),name="encoder_token_type_ids",dtype=tf.int32)
    input_attention_mask_layer = layers.Input(shape=(max_input_length,),name="encoder_attention_mask",dtype=tf.int32)
    input_length = layers.Input(shape=(1,),name="length",dtype=tf.int32)
    base_model.trainable = True
    base_model = base_model(input_ids_layer,token_type_ids=input_type_ids_layer,attention_mask=input_attention_mask_layer,return_dict=True)
    output = layers.Dropout(0.1)(base_model.last_hidden_state)
    output = layers.Dense(1024,activation="relu")(output)
    output = layers.Dropout(0.1)(output)
    output = layers.Dense(1024,activation="relu")(output)
    output = layers.Dense(3,activation="linear")(output)
    crf = CRFLayer()
    output = crf(inputs=[output,input_length])
    model = models.Model(inputs=[input_ids_layer,input_type_ids_layer,input_attention_mask_layer,input_length],outputs=output)
    model.compile(optimizer=optimizers.Adam(learning_rate=3e-5),loss=crf.loss,metrics=['accuracy'])
    return model

    
def f1(preds,trues):
    if len(trues) == 0:
        return 1. if len(preds) == 0 else 0.
    if len(preds) == 0:
        return 0.
    predictions_set = set(preds)
    gold_set = set(trues)
    nom = 2 * len(predictions_set.intersection(gold_set))
    denom = len(predictions_set) + len(gold_set)
    return float(nom)/float(denom)

def avg_f1(preds,trues):
    avg_f1_total = 0.0
    for pred,true in zip(preds,trues):
        avg_f1_total += f1(pred,true)
    return avg_f1_total/len(preds)

class F1Metric(callbacks.Callback):
    def __init__(self,inputs,labels,spans,texts,test=True):
        self.inputs = inputs
        self.spans = spans
        self.tokenizer = tokenizer
        self.texts = texts
        self.test = test

    def on_epoch_end(self, epoch, logs={}):
        preds = self.model.predict(self.inputs,verbose=0)
        indices = createIndicesForNERModel(preds,texts,tokenizer)
        f1 = avg_f1(indices,self.spans)
        if self.test:
            print()
            print("test f1 = "+str(f1))
        else:
            print()
            print("train f1 = "+str(f1))
