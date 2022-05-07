import pickle
import json
import string
import re 
import spacy


INPUT_DATA_PATH="data-0505"
BASELINE=""#"qa","ontology-qa"


def raw_data_to_standard_output():
	with open(INPUT_DATA_PATH+"/test.pkl","rb") as f:
		data=pickle.load(f)


	output=[]
	for item in data:
		ann={}
		ann["content"]=item[1]
		ann["annotations"]={}
		for a in json.loads(item[-1]):

			if a["type"] not in ann["annotations"]:
				 ann["annotations"][a["type"]]=[]
			ann["annotations"][a["type"]].append(a["text"])
		output.append(ann)

	with open(INPUT_DATA_PATH+"/"+BASELINE+"/evaluate_format_golden.pkl","wb") as f:
		pickle.dump(output,f)


def normalize_string(s):
	"""Lower text and remove punctuation, articles and extra whitespace."""
	def remove_articles(text):
		regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
		return re.sub(regex, ' ', text)
	def white_space_fix(text):
		return ' '.join(text.split())
	def remove_punc(text):
		exclude = set(string.punctuation)
		return ''.join(ch for ch in text if ch not in exclude)
	def lower(text):
		return text.lower()
	return white_space_fix(remove_articles(remove_punc(lower(s))))



def exact_match(final_pred,final_gold):
	predict_number=0
	annotation_number=0
	right_number=0
	for (cur_pred,cur_gold) in zip(final_pred,final_gold):
		for k,v in cur_pred.items():
			predict_number=predict_number+len(v)
		for k,v in cur_gold.items():
			annotation_number=annotation_number+len(v)
			for vitem in v:
				if k in cur_pred:#首先是预测有这个事件要素类型
					for temp in cur_pred[k]:#其次是抽的该事件要素类型的答案也对
						if normalize_string(temp)==normalize_string(vitem):
							right_number=right_number+1
							# print(temp.lower())
							break
	print(predict_number,annotation_number,right_number)
	precision=float(right_number)/float(predict_number)
	recall=float(right_number)/float(annotation_number)
	f=(2*float(precision)*float(recall))/(float(precision)+float(recall))
	print("EM Precision:",precision)
	print("EM Recall:",recall)
	print("EM F_score:",f)

def get_head_noun(mention,nlp):
	mention_norm = normalize_string(mention)
	head_noun = []
	noun_chunks = list(nlp(mention_norm).noun_chunks)
	for noun_chunk in noun_chunks: 
		head_noun.append(noun_chunk.root.text)
	return list(set(head_noun))

# def rough_match():
import tqdm
import tqdm
def head_noun_match(final_pred,final_gold):
	nlp = spacy.load("zh_core_web_sm-3.2.0/zh_core_web_sm/zh_core_web_sm-3.2.0")#("en_core_web_sm-3.2.0/en_core_web_sm/en_core_web_sm-3.2.0")
	predict_number=0
	annotation_number=0
	right_number=0
	progress=0
	for (cur_pred,cur_gold) in zip(final_pred,final_gold):
		# progress=progress+1
		# if progress%200==0:
		# 	print(progress)
		for k,v in cur_pred.items():
			predict_number=predict_number+len(v)
		for k,v in cur_gold.items():
			annotation_number=annotation_number+len(v)
			for vitem in v:
				# vitem_hn=get_head_noun(vitem,nlp)
				if k in cur_pred:#首先是预测有这个事件要素类型
					for temp in cur_pred[k]:#其次是抽的该事件要素类型的答案也对
						if normalize_string(temp)==normalize_string(vitem):
							right_number=right_number+1
							break
						if normalize_string(temp) in normalize_string(vitem) or normalize_string(vitem) in normalize_string(temp):
							right_number=right_number+1
							break

						# temp_hn=get_head_noun(temp,nlp)
						# if len(set(temp_hn)&set(vitem_hn))!=0:
						# 	right_number=right_number+1
						# 	break
								
	print(predict_number,annotation_number,right_number)
	precision=float(right_number)/float(predict_number)
	recall=float(right_number)/float(annotation_number)
	f=(2*float(precision)*float(recall))/(float(precision)+float(recall))
	print("HM Precision:",precision)
	print("HM Recall:",recall)
	print("HM F_score:",f)




import utils
def evaluate(predict_file,golden_file):
	
	predicts=pickle.load(open(predict_file,"rb"))
	goldens=pickle.load(open(golden_file,"rb"))
	test_length = [1 for i in range(len(goldens))]
	assert len(goldens)==len(test_length)

	final_pred=[]
	final_gold=[]
	pre_index=0
	for index in range(len(goldens)):
		cur_pred={}
		cur_pred_content=""
		for p_index in range(pre_index,pre_index+test_length[index]):
			span_annotation=predicts[p_index]
			for c in span_annotation["content"]:
				cur_pred_content+=c
			for k,v in span_annotation["annotations"].items():
				if k not in cur_pred:
					cur_pred[k]=[]
				cur_pred[k].extend(v)
		# print("############")
		# print(cur_pred)
		for k,v in cur_pred.items():
			cur_pred[k]=list(set(cur_pred[k]))
		cur_gold=goldens[index]["annotations"]
		# print(cur_gold)
		pre_index=pre_index+test_length[index]
		# print("############")
		final_pred.append(cur_pred)
		final_gold.append(cur_gold)

	exact_match(final_pred,final_gold)
	head_noun_match(final_pred,final_gold)



raw_data_to_standard_output()# get evaluate_format_golden.pkl file
evaluate(INPUT_DATA_PATH+"/"+BASELINE+"/preds4.pkl",INPUT_DATA_PATH+"/"+BASELINE+"/evaluate_format_golden.pkl")
#evaluate_format_predict.pkl









