import pickle
import json
from transformers import BertTokenizerFast
from tqdm import tqdm

INPUT_DATA_PATH="data-0505"
MODEL_NAME="bert-base-uncased"#"model/clue/roberta_chinese_base/"
tokenizer= BertTokenizerFast.from_pretrained(MODEL_NAME)
vocab = tokenizer.get_vocab()
id2vocab = dict(zip(vocab.values(), vocab.keys()))

def span2bio(record):
    #print(record,len(record))
    text = record[1]
    meta = json.loads(record[3])
    encoding = tokenizer(text, return_offsets_mapping=True, padding=False, truncation=False)
    
    words = []
    labels = []
    for input_id, offset in zip(encoding.input_ids[1:-1], encoding.offset_mapping[1:-1]):
        words.append(id2vocab[input_id])
        flag = False
        for tag in meta:
            if offset[0] == tag['start'] :
                labels.append('B-' + '-'.join(tag['type'].strip().split()))
                flag = True
                break
            elif offset[0] > tag['start']  and offset[1] <= tag['end']+1:
                labels.append('I-' + '-'.join(tag['type'].strip().split()))
                flag = True
                break
        if not flag:
            labels.append('O')
    assert len(words) == len(labels)
    words, labels = recover_word_piece(words, labels)
    return words, labels

def recover_word_piece(words, labels):
    rwords = []
    rlabels = []

    for word, label in zip(words, labels):
        if not word.startswith('##'):
            rwords.append(word)
            rlabels.append(label)
        else:
            rwords[-1] += word.strip('##')
    assert len(rwords) == len(rlabels)
    return rwords, rlabels

def write_bio(sents, labels, fn):
    with open(fn, 'w') as fout:
        for sent, label in zip(sents, labels):
            for word, l in zip(sent, label):
                fout.write(word + '\t' + l + '\n')
            fout.write('\n')

def generate_label2id(labels):
    label2id={}
    index=0
    for label in labels:
        for i in label:
            if i not in label2id:
                label2id[i]=index
                index=index+1
    with open(INPUT_DATA_PATH+"/label2id.pkl","wb") as f:
        pickle.dump(label2id,f)

def convert_fn_span_to_bio(ifn, ofn):
    with open(ifn, 'rb') as fin:
        records = pickle.load(fin)
    sents, labels = [], []
    
    for record in tqdm(records):
        words, label = span2bio(record)
        sents.append(words)
        labels.append(label)
    write_bio(sents, labels, ofn)
    return labels

def split_sentence_by_max_length(words, labels, max_lenght):
    doc = ' '.join(words)
    
    
    
    

if __name__ == '__main__':
    test_label=convert_fn_span_to_bio(INPUT_DATA_PATH+'/test.pkl', INPUT_DATA_PATH+'/test.tsv')
    dev_label=convert_fn_span_to_bio(INPUT_DATA_PATH+'/dev.pkl', INPUT_DATA_PATH+'/dev.tsv')
    train_label=convert_fn_span_to_bio(INPUT_DATA_PATH+'/train.pkl', INPUT_DATA_PATH+'/train.tsv')
    labels=test_label+dev_label+train_label
    generate_label2id(labels)
      
        
    
    
