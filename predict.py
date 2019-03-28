# coding:utf-8
'''
author:wangyi
'''
import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as kr
import thulac
from cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category#,get_word_embedding


class Predict:

    def __init__(self,stopwords_path,vocab_dir,categories_dir,save_path):

        self.thu = thulac.thulac(seg_only=True)
        self.stopwords = [line.strip() for line in open(stopwords_path).readlines()]
        categories, cat_to_id = read_category(categories_dir)
        self.id_to_cat = {v:k for k,v in cat_to_id.items()}
        words, self.word_to_id = read_vocab(vocab_dir)
        g = tf.Graph()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(graph=g,config=tf_config)
        with self.sess.as_default():
            with g.as_default():
                self.config = TCNNConfig()
                self.config.num_classes = len(cat_to_id)
                self.config.vocab_size = len(words)
                self.model = TextCNN(self.config)
                saver = tf.train.Saver()
                self.sess.run(tf.global_variables_initializer())
                saver.restore(self.sess,save_path=save_path)

    def predict(self,contents):
        contents = contents.strip()
        contents = self.get_cut_words_res(contents)
        x_pad = self.process_file(contents,self.config.seq_length)
        with self.sess.as_default():
            with self.sess.graph.as_default():
                pred_prob = self.sess.run(self.model.pred_prob,feed_dict={self.model.input_x:x_pad,self.model.keep_prob:1.0})[0]
                res = []
                for i in range(len(pred_prob)):
                    if pred_prob[i] >= 0.5:
                        res.append({'label':self.id_to_cat[i],'prob':np.round(float(pred_prob[i]),2)})
                if len(res) < 1:
                    res.append({'label':self.id_to_cat[np.argmax(pred_prob)],'prob':np.round(float(pred_prob[np.argmax(pred_prob)]),2)})
        return res

    def process_file(self,contents,max_length=400):
        """将文件转换为id表示"""

        data_id = []
        for i in range(len(contents)):
            data_id.append([self.word_to_id[x] for x in contents[i] if x in self.word_to_id])
        # 使用keras提供的pad_sequences来将文本pad为固定长度
        x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length, padding='post', truncating='post')
        return x_pad

    def get_cut_words_res(self,content):
        content = self.thu.cut(content, text=True).split(' ')
        words = []
        for word in content:
            if word not in self.stopwords:
                words.append(word)
        return [words]

if __name__ == '__main__':

    predict = Predict('/home/abc/pySpace/disciplinary_action/datasets/stopwords.txt',
                      '/home/abc/pySpace/disciplinary_action/datasets/vocab.txt',
                      '/home/abc/pySpace/disciplinary_action/datasets/categories.txt',
                      '/home/abc/pySpace/disciplinary_action/checkpoints/best_validation')
    contents = '伽师县食品药品监督管理局药品监督检查室干部艾力亚尔·木塔力甫收受药店经营人员礼金礼品，2016年1月受到党内严重警告处分。'
    res = predict.predict(contents)
    print(res)