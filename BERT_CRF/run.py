import pandas as pd
import json
import gc
import tensorflow as tf
from tensorflow.keras import *
from sklearn.model_selection import KFold
from transformers import TFBertModel,BertTokenizer
from ner import createNEROutputs, createToxicModelWithGivenBaseModel, createInputForNER, createIndicesForNERModel, avg_f1
from crf import *

test_set = pd.read_csv("fold_1.csv")
test_set['spans'] = test_set['spans'].apply(lambda x : json.loads(x))

train_set=pd.read_csv("fold_2.csv" )
train_set['spans'] = train_set['spans'].apply(lambda x : json.loads(x))

for i in range(3):
    train=pd.read_csv("fold_%s.csv" %(i+3))
    train['spans'] = train['spans'].apply(lambda x : json.loads(x))
    train_set=train_set.append(train,ignore_index=True)
    
toxic_span_dataset = train_set
toxic_span_dataset['text'] = toxic_span_dataset['text'].apply(lambda x : x.lower())



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
max_length=400
#make train set
texts = toxic_span_dataset['text'].to_numpy()
targets = createNEROutputs(texts,toxic_span_dataset['spans'],max_length,tokenizer)
all_spans = toxic_span_dataset['spans'].to_numpy()
result_test = []
result_train = []
kf = KFold(n_splits=5,shuffle=True)
train_test_indices = []
for train_index,test_index in kf.split(texts):
    train_test_indices.append((train_index,test_index))
    

#random training
train_index,test_index = train_test_indices.pop()
x_train , x_test = list(texts[train_index]) , list(texts[test_index])
y_train , y_test = targets[train_index] , targets[test_index]
model = None
base_model = None
gc.collect()
tf.keras.backend.clear_session()
base_model = TFBertModel.from_pretrained('bert-base-uncased')
model = createToxicModelWithGivenBaseModel(max_length,base_model)
train_data = createInputForNER(x_train,max_length,tokenizer)
test_data = createInputForNER(x_test,max_length,tokenizer)
spans_test = all_spans[test_index]
spans_train = all_spans[train_index]
model.fit(train_data,y_train,batch_size=16,epochs=2,callbacks=[callbacks.ModelCheckpoint("working",save_weights_only=True)])
preds = model.predict(test_data)
indices = createIndicesForNERModel(preds,x_test,tokenizer)
f1_toxic = avg_f1(indices,spans_test)
print("test F1 = %f"%(f1_toxic))
result_test.append(f1_toxic)
preds = model.predict(train_data)
indices = createIndicesForNERModel(preds,x_train,tokenizer)
f1_toxic = avg_f1(indices,spans_train)
print("train F1 = %f"%(f1_toxic))
result_train.append(f1_toxic)

#make test set
test_set['text']=test_set['text'].apply(lambda x : x.lower())
test_texts=test_set['text'].to_numpy()
x_test_1=list(test_texts)
test_data_1 = createInputForNER(x_test_1,max_length,tokenizer)

#make prediction on test
preds_1 = model.predict(test_data_1)
indices = createIndicesForNERModel(preds_1,x_test_1,tokenizer)

