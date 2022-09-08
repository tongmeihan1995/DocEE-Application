import pickle
import json
import torch
import os
import numpy as np
from transformers import BertTokenizerFast, AutoModelForQuestionAnswering


#tokenizer = BertTokenizerFast.from_pretrained('dslim/bert-base-NER')
# MODEL_NAME="bert-base-chinese"
# tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)
from transformers import LongformerTokenizerFast, LongformerForQuestionAnswering

tokenizer = LongformerTokenizerFast.from_pretrained("valhalla/longformer-base-4096-finetuned-squadv1")

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7,8"

def read_from_pickle(fn):
    contexts = []
    questions = []
    answers = []

    with open(fn, 'rb') as fin:
        dataset = pickle.load(fin)

    for record in dataset:
        text = record[1]
        labels = json.loads(record[-1])
        for l in labels:
            ans = {}
            contexts.append(text)
            questions.append(l['type'])
            # ans['text'] = l['text']
            # ans['answer_start'] = l['start']
            # ans['answer_end'] = l['end']
            answers.append(ans)
    return contexts, questions, answers

# train_contexts, train_questions, train_answers = read_from_pickle('data-0505/train.pkl')
test_contexts, test_questions, test_answers = read_from_pickle('data-0505/test_after_cl.pkl')


# train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
test_encodings = tokenizer(test_contexts, test_questions, truncation=True, padding=True)

# def add_token_positions(encodings, answers):
#     start_positions = []
#     end_positions = []
#     for i in range(len(answers)):
#         start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
#         end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))
#         # if None, the answer passage has been truncated
#         if start_positions[-1] is None:
#             start_positions[-1] = tokenizer.model_max_length
#         if end_positions[-1] is None:
#             end_positions[-1] = tokenizer.model_max_length
#     encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# # add_token_positions(train_encodings, train_answers)
# add_token_positions(test_encodings, test_answers)

    
class SquadDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        print(len(self.encodings.input_ids))
        return len(self.encodings.input_ids)

# train_dataset = SquadDataset(train_encodings)
test_dataset = SquadDataset(test_encodings)

model = AutoModelForQuestionAnswering.from_pretrained("results/checkpoint-1000")#("bert-base-uncased")
'''
from torch.utils.data import DataLoader
from transformers import AdamW

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

model.to(device)
model.train()

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

optim = AdamW(model.parameters(), lr=5e-5)

steps = 0
for epoch in range(3):
    print('epoch:%d'%epoch)
    for batch in train_loader:
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions)
        loss = outputs[0]
        if steps % 100 == 0:
            print('steps:%d'%steps)
            print(loss)
        loss.backward()
        optim.step()
        steps += 1

print(model.eval())
'''
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    per_device_eval_batch_size=32,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    learning_rate=5e-5,
    eval_accumulation_steps=6
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
)
#  train_dataset=train_dataset,         # training dataset
# eval_dataset=test_dataset,             # evaluation dataset

# trainer.train()
# print(trainer.evaluate())
res = trainer.predict(test_dataset)

starts = np.argmax(res[0][0], axis = -1)
ends = np.argmax(res[0][1], axis = -1)

# print(starts[:100])
# print(ends[:100])
# print(res[1][0][:100])
# print(res[1][1][:100])
# print(len(starts))
# num = 0

# for ps, pe, start, end in zip(starts, ends, res[1][0], res[1][1]):
#     if ps == start and pe == end:
#         num += 1

# print(num)

with open('data-0505/test_after_cl.pkl', 'rb') as fin:
    test = pickle.load(fin)

preds = []

ind = 0
for line in test:
    cr = {}
    # cr['content'] = line[1]
    labels = json.loads(line[-1])
    annotations = {}
    for label in labels:
        if not label['type'] in annotations:
            annotations[label['type']] = tokenizer.convert_ids_to_tokens(test_encodings['input_ids'][ind][starts[ind]: ends[ind]+ 1])
            tr = ''
            for i, t in enumerate(annotations[label['type']]):
                #if not t.startswith('##'):
                if t.startswith('Ġ'):
                    if i != 0:
                        tr += ' '
                    tr += t
                else:
                    tr += t
            tr = tr.replace('Ġ', '')
            annotations[label['type']] = [tr]
        ind += 1
    cr['annotations'] = annotations
    cr['golden'] = labels
    preds.append(cr)

print(preds[:3])
# with open('data-0505/preds4.pkl', 'wb') as fout:
#     pickle.dump(preds, fout)
        








