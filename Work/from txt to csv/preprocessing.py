import pandas as pd
import numpy as np

class read_txt:
    def __init__(self, path = 'sample.txt', length_lst = None):
        self.path = path
        self.length_lst = length_lst
        self.data = []
     
    def preprocess(self):
        data = []
        with open(self.path) as txt:
            lines = txt.readlines()
            for line in lines:
                letter = ''
                for letter_ in line:
                    if letter_==' ' or letter_=='\t' or letter_=='\n':
                        pass
                    else:
                        letter = letter+letter_
                data.append(letter)
        return data

    def slice(self, sentence):
        # if len(sentence)!=len(self.length_lst):
        #     raise ValueError('입력받은 length_lst와 문장의 길이가 일치하지 않음')
        data = []
        start, end = 0, self.length_lst[0]
        for i in range(len(self.length_lst)):
            if i<len(self.length_lst)-1:
                word = sentence[start:end]
                start += self.length_lst[i]
                end += self.length_lst[i+1]
            else:
                word = sentence[start:]
            data.append(word)
        return data

    def save_csv(self, save_name = 'sample'):
        data = [self.slice(sentence) for sentence in self.preprocess()]
        data = pd.DataFrame(np.array(data))
        data.to_csv('{}.csv'.format(save_name), index = False)



path = 'sample.txt'
length_lst = [8,8,8,2]
save_name = 'sample'

func = read_txt(path = path, length_lst = length_lst)
func.save_csv(save_name = save_name)
