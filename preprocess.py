# coding:utf-8
'''
author:wangyi
'''

import pandas as pd
import numpy as np
import pickle
import random
import thulac
import re


def cut_words(inf,stopwords_path,train_out,val_out,test_out):
    thu = thulac.thulac(seg_only=True,filt=True)
    stopwords = [line.strip() for line in open(stopwords_path).readlines()]
    #label2id = {line.split('\t')[1].strip():line.split('\t')[0] for line in open('./datasets/labels.txt').readlines()}
    #pickle.dump(label2id,open('./datasets/label2id.pkl', 'wb'))
    label2id = pickle.load(open('./datasets/label2id.pkl','rb'))
    train_writer = open(train_out,'w')
    val_writer = open(val_out,'w')
    test_writer = open(test_out,'w')
    link = {}
    with open(inf,encoding='utf-8') as f:
        lines = f.readlines()
        random.shuffle(lines)
        for i,line in enumerate(lines):
            content = line.split('\t')[0]
            src = content.strip()
            content = re.sub('\n','',content)
            label = '\t'.join(line.split('\t')[1].strip().split('#'))
            content = thu.cut(content, text=True).split(' ')
            words = []
            for word in content:
                if word not in stopwords:
                    words.append(word)
            content = ' '.join(words)
            link[content] = src
            data = label+'\t'+content.strip()+'\n'
            if i<int(len(lines)*0.8):
                train_writer.write(data)
                train_writer.flush()
            elif i<int(len(lines)*0.9):
                val_writer.write(data)
                val_writer.flush()
            else:
                test_writer.write(data)
                test_writer.flush()
            print(i)
        pickle.dump(link,open('./datasets/link.pkl','wb'))


def statics_length():
    lengths = []
    for line in open('./datasets/train.txt').readlines():
        lengths.append(len(line.strip().split('\t')[-1].split(' ')))
    for line in open('./datasets/val.txt').readlines():
        lengths.append(len(line.strip().split('\t')[-1].split(' ')))
    for line in open('./datasets/test.txt').readlines():
        lengths.append(len(line.strip().split('\t')[-1].split(' ')))
    print(np.percentile(lengths,83))


def create_labels():
    label2id = pickle.load(open('./datasets/label2id.pkl','rb'))
    writer = open('./datasets/categories.txt','w')
    for k in label2id.keys():
        writer.write(k.strip()+'\n')


def link_src_cut():
    # thu = thulac.thulac(seg_only=True, filt=True)
    # stopwords = [line.strip() for line in open('./datasets/stopwords.txt').readlines()]
    tests = []
    res = []
    # link = {}
    with open('./datasets/test.txt') as f:
        for line in f.readlines():
            tests.append(line.split('\t')[-1])
    # with open('./datasets/wjlabel_0313.csv') as f:
    #     lines = f.readlines()
    #     for i, line in enumerate(lines):
    #         src = line.split('\t')[0]
    #         content = re.sub('\n', '', src)
    #         content = thu.cut(content, text=True).split(' ')
    #         words = []
    #         for word in content:
    #             if word not in stopwords:
    #                 words.append(word)
    #         content = ' '.join(words)
    #         link[content] = src
    #         print(i)
    #     pickle.dump(link,open('./datasets/link.pkl','wb'))
    link = pickle.load(open('./datasets/link.pkl','rb'))
    for i,t in enumerate(tests):
        if t.strip() in link.keys():
            res.append([link[t.strip()],t])
        print('t',i)
    res = pd.DataFrame(res)
    res.to_excel('./datasets/link.xls',index=False,encoding='utf-8')


def create_predict_set():
    thu = thulac.thulac(seg_only=True, filt=True)
    stopwords = [line.strip() for line in open('./datasets/stopwords.txt').readlines()]
    out = open('./datasets/predict.txt','w',encoding='utf-8')
    with open('./datasets/noLabelData_0319.txt') as f:
        lines = f.readlines()
        for i,line in enumerate(lines):
            out.write('收受请托人财物行为'+'\t'+get_cut_words_res(thu,line,stopwords).strip()+'\n')
            out.flush()
            print(i)

def get_predict_src():
    with open('./datasets/noLabelData_0319.txt') as f:
        lines = f.readlines()
        lines = [[line.strip()] for line in lines]
        lines = pd.DataFrame(lines)
        lines.to_excel('./datasets/predict_src.xlsx',index=False,encoding='utf-8')

def get_cut_words_res(thu,content,stopwords):
    content = thu.cut(content, text=True).split(' ')
    words = []
    for word in content:
        if word not in stopwords:
            words.append(word)
    content = ' '.join(words)
    return content




if __name__ == '__main__':

    #cut_words('./datasets/wjal_labeled_0322.txt','./datasets/stopwords.txt','./datasets/train.txt','./datasets/val.txt','./datasets/test.txt')

    statics_length()

    #create_labels()

    #link_src_cut()

    #create_predict_set()

    #get_predict_src()

