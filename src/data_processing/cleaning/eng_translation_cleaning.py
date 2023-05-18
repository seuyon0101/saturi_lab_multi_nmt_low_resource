import os
path = os.getenv('HOME')
print(path)

if __name__ == '__main__' :

	import pandas as pd
	import tqdm
	import re
	from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
	from torch.utils.data import DataLoader

	# load data

	df = pd.read_csv('dataset_full_v1.csv') # data found on google drive 'raw' directory

	#data cleaning

	df['text']=df['text'].apply(lambda x : re.sub('[\(\)]','',str(x))) # remove parenthesis
	df['text']=df['text'].apply(lambda x : re.sub('(\&+[a-zA-Zㄱ-힣\&]+)','',str(x))) # remove words like &&name, &company name, etc.
	df['text']=df['text'].apply(lambda x : re.sub('(\@+[ㄱ-힣\@]+)','',str(x))) # remove words like &&name, &company name, etc.
	df['text']=df['text'].apply(lambda x : re.sub('(x+[a-zㄱ-힣]+)','',str(x))) # remove words like xx하면, xxxx, etc
	df['text']=df['text'].apply(lambda x : re.sub('(\-+[a-zㄱ-힣\-]+)','',str(x))) # remove words like --하면, -시사- etc
	df['text']=df['text'].apply(lambda x : re.sub(r'(.+)\1{2,}',r'\1',str(x))) # remove repeating words
	df['text']=df['text'].apply(lambda x : re.sub('([ㄱ-힣]+)\~$',r'\1.',str(x))) # sub words ending in ~ to . ex 그럴거야~ 그럴거야.
	df['text']=df['text'].apply(lambda x : re.sub('([ㄱ-힣]+)\~','',str(x))) # remove expressions like 어~ 아~ 으~ 오~

	df['dial']=df['dial'].apply(lambda x : re.sub('[\(\)]','',str(x)))
	df['dial']=df['dial'].apply(lambda x : re.sub('(\&+[a-zA-Zㄱ-힣\&]+)','',str(x)))
	df['dial']=df['dial'].apply(lambda x : re.sub('(\@+[ㄱ-힣\@]+)','',str(x))) # remove words like @이름,@주소,@상호명, etc.
	df['dial']=df['dial'].apply(lambda x : re.sub('(x+[a-zㄱ-힣]+)','',str(x))) # remove words like xx하면, xxxx, etc
	df['dial']=df['dial'].apply(lambda x : re.sub('(\-+[a-zㄱ-힣\-\&]+)','',str(x))) # remove words like --하면, -시사- etc
	df['dial']=df['dial'].apply(lambda x : re.sub(r'(.+)\1{2,}',r'\1',str(x))) # remove repeating words
	df['dial']=df['dial'].apply(lambda x : re.sub('([ㄱ-힣]+)\~$',r'\1.',str(x))) # sub words ending in ~ to . ex 그럴거야~ 그럴거야.
	df['dial']=df['dial'].apply(lambda x : re.sub('([ㄱ-힣]+)\~','',str(x))) # remove expressions like 어~ 아~ 으~ 오~

	#remove from training

	df.loc[df['text'].str.contains('/'),'rm'] = 1 #containing mislabelled data
	df['len'] = df['text'].apply(lambda x: len(x))
	df.loc[df['len'] < 2,'rm'] = 1 #length is less than 2
	df.drop_duplicates(subset='dial', inplace=True) # drop duplicated senteces in dial
	clean_df = df.loc[df['rm']!=1,['text','dial','reg','pair']]

	# translation pipeline

	ckpt = 'circulus/kobart-trans-ko-en-v2'
	tokenizer = AutoTokenizer.from_pretrained(ckpt)
	model = AutoModelForSeq2SeqLM.from_pretrained(ckpt)
	translator = pipeline("translation", model=ckpt)

	text = clean_df.text.values.tolist()
	dataloader = DataLoader(text,  batch_size=512)
    
    #save translations
	outputs = []
	for n,i in enumerate(tqdm.tqdm(dataloader)) :
	    inputs = tokenizer(i, return_tensors="pt", padding=True).input_ids
	    out = model.generate(inputs,num_beams=3, max_new_tokens=60) # do_sample=True, top_k=30, top_p=0.95,
	    for j in out :
	        result = tokenizer.decode(j, skip_special_tokens=True)
	        outputs.append(result)
	    
	    if n % 100 == 0 : #save every 100th batch
	        with open('translated.txt','w') as f :
	            for sen in outputs :
	                f.write(sen+'\n')
   
    # save all remaining outputs                 
	with open('translated.txt','r') as f :
        raw = f.read().splitlines()

    #add translated english list as new column
	clean_df['eng'] = raw
    
    # export to csv
	clean_df.to_csv('translated_train_data.csv')