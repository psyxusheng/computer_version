import numpy as np
import json
import HyperParameter as HP


ml = max_length = HP.time_steps


class Vocab():
    """
    set 'unk' as index 1
    'pad' as index 0
    """
    def __init__(self,
        filename,):
        self._read_dict(filename)
        self._unk = self._word2id['u']
        self._pad = self._word2id['p']
        
    def _read_dict(self,filename):
        self._word2id = json.load(open(filename,'r',encoding='utf8'))
        

    def sentence2id(self,sentence):
        word_ids = [self._word2id.get(word,self._unk) for word\
                     in list(sentence)]
        return word_ids
    
    @property
    def unkid(self):
        return self._unk

    @property
    def padid(self):
        return self._pad
    @property
    def size(self):
        return self._word2id.__len__()
    
    def check(self,w):
        return self._word2id.get(w,'not found')


class Category():
    def __init__(self,filename):
        self._cate2id = {}
        with open(filename,'r',encoding='uutf8') as f:
            for line in f:
                cate = line.strip()
                idx = self._cate2id.__len__()
                self._cate2id[cate]=idx
    def category2id(self,cateName):
        if not cateName in self._cate2id:
            raise Exception('%s not in category list')
        return self._cate2id[cateName]
    @property
    def size(self,):
        return self._cate2id.__len__()



class TextDataSet():
    def __init__(self,textFileName,
    vocab,
    #cates,
    max_length=ml):
        
        self._vocab = vocab
        #self._cates = cates
        self._max_length = ml
        self._cursor = 0
        self._parse_file(textFileName)
        self.random_shuffle()
    
    def _parse_file(self,textFileName):
        print('loading data...')
        Xs , Ys = [],[]
        with open(textFileName,'r',encoding='utf8') as f:
            for line in f:
                label,content = line.strip().split('\t')
                #label_id = self._cates.category2id(label)
                words_id = self._vocab.sentence2id(content)
                words_id = words_id[:self._max_length]
                padding_num = self._max_length-words_id.__len__()
                words_id = words_id + [self._vocab.padid]*padding_num
                Xs.append(words_id)
                Ys.append(int(label))
        self._Xs = np.array(Xs,dtype=np.int32)
        self._Ys = np.array(Ys,dtype=np.int32)
        self._size = self._Xs.shape[0]
        self._classes = np.bincount(self._Ys).shape[0]
        print('loaded    ...')

    def random_shuffle(self,):
        pos = np.random.permutation(self._size)
        self._Xs = self._Xs[pos]
        self._Ys = self._Ys[pos]

    def next_batch(self,batch_size=100):
        end_cursor = self._cursor+batch_size
        if end_cursor > self._size:
            self._cursor = 0
            self.random_shuffle()
            end_cursor = batch_size
        if end_cursor > self._size:
            raise Exception('Btach_Size too big')
        batch_inputs = self._Xs[self._cursor:end_cursor]
        batch_outputs = self._Ys[self._cursor:end_cursor]
        self._cursor  = end_cursor
        return batch_inputs,batch_outputs
    def next_batch_balanced(self,batch_size):
        assert batch_size % self._classes == 0
        each_num = batch_size // self._classes
        xs,ys = [],[]
        for i in range(self._classes):
            pos = self._Ys == i
            curr_X = self._Xs[pos,:]
            curr_Y = self._Ys[pos]
            size = curr_X.shape[0]
            ridx = np.random.choice(size,size=each_num,replace=False)
            xs.append(curr_X[ridx,:])
            ys.append(curr_Y[ridx])
        inputs = np.concatenate(xs,axis=0)
        targets = np.concatenate(ys,axis=0)
        pos = np.random.permutation(batch_size)
        inputs_ = inputs[pos,:]
        targets_ = targets[pos]
        return inputs_,targets_
    @property
    def size(self):
        return self._size
    @property
    def X(self):
        return self._Xs
    @property
    def Y(self):
        return self._Ys
    @property
    def cls(self):
        return self._classes


if __name__ == '__main__':
    vocab = Vocab('vocab.json')
    trainText = TextDataSet('test.txt',vocab)