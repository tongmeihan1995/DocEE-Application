import os
import pickle
import torch
import numpy as np
import json
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score

MODEL_NAME="bert-base-uncased"#"model/roberta-base"#"model/distilbert-base-uncased"#"model/albert-base-v2"#"model/bert-base-cased"
#MODEL_NAME="results/checkpoint-1500"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#"1,2,3,4,5,6,7,8"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

base_dir = "data-0505"
with open(os.path.join(base_dir, 'test.pkl'), 'rb') as fin:
    test = pickle.load(fin)


def construct_label2id(base_dir,train,dev,test):
    path=os.path.join(base_dir, 'label2id_classify')
    if os.path.exists(path):
        label2id=pickle.load(open(path,"rb"))
    else:
        print("construct label2id file")
        whole=train+dev+test
        labels=[]
        for f in whole:
            # print(json.loads(f[3]))
            labels.extend([i["type"] for i in json.loads(f[3])])
        labels=list(set(labels))
        label2id={}
        for i,l in enumerate(labels):
            label2id[l]=i
        print("num_labels:",len(label2id.keys()))
        print(label2id)
        pickle.dump(label2id,open(path,"wb"))
    return label2id

label2id=construct_label2id(base_dir,[],[],[])

def get_sents_and_labels(dataset,num_labels):
    sents = [str(f[0])+" "+str(f[1]) for f in dataset]
    labels=[]
    for f in dataset:
        label=[0 for i in range(num_labels)]
    #     for i in json.loads(f[3]):
    #         label[label2id[i["type"]]]=1
    #     # label=[label2id[i["type"]] for i in json.loads(f[3])]
        labels.append(label)
    # for i in range(1):
    #     print(sents[i])
    #     print(labels[i])
    return sents, labels

num_labels=len(label2id.keys())
test_sents, test_labels = get_sents_and_labels(test,num_labels)


test_encodings = tokenizer(test_sents, truncation=True, padding=True, max_length=512)

class EventDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        return len(self.labels)

test_dataset = EventDataset(test_encodings, test_labels)
print("test size:",len(test_dataset))

# from transformers import BertForSequenceClassification, Trainer, TrainingArguments
# from transformers import DistilBertForSequenceClassification,RobertaForSequenceClassification,AlbertForSequenceClassification
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_metric

training_args = TrainingArguments(
    output_dir='./results_cl',          # output directory
    num_train_epochs=8,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=8,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
)
#model:DistilBert,Roberta,Albert,Bert  MODEL_NAME
model = AutoModelForSequenceClassification.from_pretrained("results_cl/checkpoint-500", num_labels=len(label2id.keys()), problem_type="multi_label_classification")

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
   
)
 # train_dataset=train_dataset,         # training dataset
 # eval_dataset=test_dataset,             # evaluation datase
 # compute_metrics=compute_metrics,

# trainer.train()
# trainer.evaluate()
# test_encodings_2 = tokenizer(test_sents, truncation=True, padding=True, max_length=512,return_tensors="pt")
# outputs=model(input_ids=test_encodings["input_ids"])
# print(outputs)
# print(outputs.get("logits"))
import numpy as np

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


res = trainer.predict(test_dataset)
# print(res)
preds=[]
preds_raw = res[0]#np.argmax(res[0], axis = -1)
print(preds_raw.shape)
assert preds_raw.shape[1]==num_labels
for item in preds_raw:
    item=sigmoid(item)
    # print(item)
    item=list(item)
    now=[0 for i in range(num_labels)]
    for index in range(len(item)):
        if item[index]>0.3:
            now[index]=1
    preds.append(np.array(now))
preds=np.array(preds)
# print(test_labels[0])
print(preds[0])


outputs=[]
id2label={y:x for x,y in label2id.items()}
for ti in range(len(test)):
    title=test[ti][0]
    content=test[ti][1]
    cur_ann=[]
    for index in range(len(preds[ti])):
        if preds[ti][index]==1:
            cur_ann.append({"type":id2label[index]})
    outputs.append([title,content,"",json.dumps(cur_ann)])

with open("data-0505/test_after_cl.pkl","wb") as f:
    pickle.dump(outputs,f)

# accuracy = accuracy_score(test_labels, preds)
# print("acc:",accuracy)
# precision = precision_score(test_labels, preds, average="micro")
# recall = recall_score(test_labels, preds, average="micro")
# f = f1_score(test_labels, preds, average="micro")
# print(precision, recall, f)



