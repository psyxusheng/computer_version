# -*-coding:utf-8 -*-
import zhon.hanzi
import string
import jieba
import re,os
from collections import Counter
import json

data_folder = 'd:/data/statusEMO'
save_file = 'weibo.txt'
regEmo = re.compile("(\[[a-zA-Z\u4e00-\u9fa5]{1,4}\])")

regTopic = re.compile('#[^#]+#')
regUsername = re.compile("@[\u4e00-\u9fa5a-zA-Z0-9_-]{2,30}")
regChinese = re.compile('[^\u4e00-\u9fa5]')

def clean(line):
    line = regEmo.subn(" ",line)[0]
    line = regTopic.subn(" ",line)[0]
    line = regUsername.subn(" ",line)[0]
    line = regChinese.subn(" ",line)[0]
    return line

    

fs = open(save_file,'w',encoding='utf8')

cates = dict()
corpus = []

for fn in os.listdir(data_folder):
    for line in open(os.path.join(data_folder,fn),'r',encoding='utf8'):
        line = line.strip()
        emos = regEmo.findall(line)
        for emo in emos:
            cates[emo] = cates.get(emo,0)+1
        if emos ==[]:
            continue
        else:
            pure = clean(line)
            corpus.append([emos,pure])
"""       
cates = sorted(cates.items(),key=lambda x:x[1],reverse=True)[:90]

fo = open('cates.txt','w',encoding='utf8')
for cat,cnt in cates:
    fo.write(cat+'\t'+str(cnt)+'\n')
fo.close()          
"""

#convert emojis to label
cates2label = {}
for line in open('labels.txt','r',encoding='utf8'):
    cat,cnt,catName,label = line.strip().split('\t')
    cates2label[cat] = label
    
for i in range(corpus.__len__()):
    emos,line = corpus.pop(-1)
    if line.strip()=="":
        continue
    tmp_lbs = {}
    for emo in emos:
        if emo in cates2label:
            tmp_lbs[emo] = tmp_lbs.get(emo,0)+1
    if tmp_lbs.__len__()==0:
        continue
    elif tmp_lbs.__len__()==1:
        for k in tmp_lbs:
            pass
        emotion = k
    else:
        emotion = sorted(tmp_lbs.items(),key=lambda x:x[1],reverse=True)[0][0]
    ct = cates2label[emotion]
    
    fs.write(str(ct)+'\t'+line+'\n')

fs.close()




#split file to train and test
#build vocab and labels file

vocab = Counter()

for line in open('weibo.txt','r',encoding='utf8'):
    lb,sentence = line.strip().split('\t')
    words = list(sentence)
    vocab.update(words)
    
vocab = vocab.most_common()[:3498]

vocab_saved = {'p':0,'u':1}
for k,c in vocab:
    vocab_saved[k] =  vocab_saved.__len__()
    
json.dump(vocab_saved,open('vocab.json','w',encoding='utf8'),ensure_ascii=False)



numLine =0
for line in open('weibo.txt','rb'):
    numLine +=1

numTrain = int(numLine*0.9)

ftrain = open('train.txt','w',encoding='utf8')
ftest = open('test.txt','w',encoding='utf8')

curr_fo = ftrain
curr_line = 0
for line in open('weibo.txt','r',encoding='utf8'):
    lb,content = line.strip().split('\t')
    content = re.subn(" +",'p',content.strip())[0]
    curr_fo.write(lb.strip()+'\t'+content+'\n')
    curr_line +=1
    if curr_line>=numTrain:
        curr_fo = ftest

ftrain.close()
ftest.close()