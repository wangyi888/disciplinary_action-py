# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
import pickle
from datetime import timedelta
from gensim.models import KeyedVectors
import datetime
import numpy as np
import tensorflow as tf
from sklearn import metrics
from sklearn.utils.multiclass import unique_labels
from cnn_model import TCNNConfig, TextCNN
from data_loader import read_vocab, read_category, batch_iter, process_file, build_vocab,read_file#,get_word_embedding
import pandas as pd
from collections import defaultdict
base_dir = './datasets'
#base_dir = '/home/abc/pySpace/disciplinary_action/datasets'
train_dir = os.path.join(base_dir, 'train.txt')
#train_dir = os.path.join(base_dir, 'example.txt')
test_dir = os.path.join(base_dir, 'test.txt')
#test_dir = os.path.join(base_dir, 'test_example.txt')
val_dir = os.path.join(base_dir, 'val.txt')
#val_dir = os.path.join(base_dir, 'example.txt')
vocab_dir = os.path.join(base_dir, 'vocab.txt')
categories_dir = os.path.join(base_dir,'categories.txt')
w2v_path = os.path.join(base_dir,'w2v.bin')
#save_dir = '/home/abc/pySpace/disciplinary_action/checkpoints'
save_dir = './checkpoints'
save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_real_seq_length(x_batch):
    real_length = []
    for x in x_batch:
        x = list(x)
        if 0 in x:
            real_length.append(x.index(0))
        else:
            real_length.append(len(x))
    return np.asarray(real_length)




def feed_data(x_batch, y_batch, keep_prob,istraining=None):
    a = get_real_seq_length(x_batch)
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
       # model.istraining:istraining
        model.batch_size:len(x_batch),
        model.real_length:get_real_seq_length(x_batch)
    }
    return feed_dict


def get_acc(pred_prob_train,y_batch):
    real_cls, pred_cls = [], []
    for p in pred_prob_train:
        m = []
        for k in range(len(p)):
            if p[k] >= 0.5:
                m.append(k)
        if len(m) < 1:
            m.append(np.argmax(p))
        pred_cls.append(m)
    for r_y in y_batch:
        m = []
        for k in range(len(r_y)):
            if r_y[k] >= 0.5:
                m.append(k)
        real_cls.append(m)
    T = 0
    for k in range(len(y_batch)):
        if real_cls[k] == pred_cls[k]:
            T += 1
    train_acc = T / len(y_batch)
    return train_acc


def evaluate(sess, x_, y_):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, config.batch_size)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        
        batch_len = len(x_batch)
       # print('什么情况',x_batch,y_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0,False)
        loss,  pred_prob= sess.run([model.loss, model.pred_prob], feed_dict=feed_dict)
        total_loss += loss * batch_len
        acc = get_acc(pred_prob,y_batch)
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def get_embeddings(w2v_path,vocab_path,word_to_id):

    model = KeyedVectors.load_word2vec_format(w2v_path)
    vocabs = [word.strip() for word in open(vocab_path).readlines()]
    embeddings = np.zeros(shape=[len(vocabs),model.vector_size])
    for v in vocabs:
        if v in model.vocab:
            embeddings[word_to_id[v]] = model[v]
        else:
            embeddings[word_to_id[v]] = np.random.uniform(-1,1,size=model.vector_size)
    return embeddings






def train():
    print("Configuring TensorBoard and Saver...")
    # 配置 Tensorboard，重新训练时，请将tensorboard文件夹删除，不然图会覆盖
    tensorboard_dir = 'tensorboard/textcnn'
    if not os.path.exists(tensorboard_dir):
        os.makedirs(tensorboard_dir)

    tf.summary.scalar("loss", model.loss)
    #tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)


    print("Loading training and validation data...")
    # 载入训练集与验证集
    start_time = time.time()
    x_train, y_train = process_file(train_dir, word_to_id, cat_to_id, config.seq_length)
    x_val, y_val = process_file(val_dir, word_to_id, cat_to_id, config.seq_length)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    # 创建session
    session = tf.Session(config=tf_config)
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)
    # 配置 Saver
    saver = tf.train.Saver(tf.global_variables())
    if not os.path.exists(save_dir):
         os.makedirs(save_dir)
    else:
         saver.restore(session,save_path=save_path)

    print('Training and evaluating...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.num_epochs):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(x_train, y_train, config.batch_size)
        for x_batch, y_batch in batch_train:
            feed_dict = feed_data(x_batch, y_batch, config.dropout_keep_prob,True)
            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            #ot =  session.run(model.rnn_output,feed_dict=feed_dict)
            #aot = session.run(model.att_out,feed_dict=feed_dict)
            #pools = session.run(model.pools,feed_dict=feed_dict)
            #pools2 = session.run(model.pool2, feed_dict=feed_dict)
            #alls = session.run(model.all,feed_dict=feed_dict)
            #convs = session.run(model.convs,feed_dict=feed_dict)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
               # feed_dict[model.istraining] = False

                loss_train, pred_prob_train = session.run([model.loss, model.pred_prob], feed_dict=feed_dict)
                train_acc = get_acc(pred_prob_train,y_batch)

                loss_val, acc_val = evaluate(session, x_val, y_val)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=save_path)
                    improved_str = '*'
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, train_acc, loss_val, acc_val, time_dif, improved_str))


            total_batch += 1

            if total_batch - last_improved > require_improvement:
                config.decay_steps = total_batch+1
                config.decay_rate = config.decay_rate*0.1
                # 验证集正确率长期不提升，提前结束训练
                #print("No optimization for a long time, auto-stopping...")
                #flag = True
                #break  # 跳出循环
        #if flag:  # 同上
        #    break


def test():
    print("Loading test data...")
    start_time = time.time()
    contents = read_file(test_dir)[0]
    x_test, y_test = process_file(test_dir, word_to_id, cat_to_id, config.seq_length)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess=session, save_path=save_path)  # 读取保存的模型

    print('Testing...')
    loss_test, acc_test = evaluate(session, x_test, y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(loss_test, acc_test))

    batch_size = 128
    data_len = len(x_test)
    num_batch = int((data_len - 1) / batch_size) + 1

    y_test_cls,y_pred_cls = [],[]
    for r_y in y_test:
        m = []
        for k in range(len(r_y)):
            if r_y[k] >= 0.5:
              m.append(k)
        y_test_cls.append(m)
    #y_test_cls = sorted(y_test_cls,reverse=True)

    #y_test_cls = np.argmax(y_test, 1)
    #y_pred_cls = np.zeros(shape=len(x_test), dtype=np.int32)  # 保存预测结果
    pred_prob = np.zeros(shape=[len(x_test),config.num_classes],dtype=np.float32)
    #sorted_pred_prob = np.zeros(shape=[len(x_test),config.num_classes],dtype=np.float32)
    for i in range(num_batch):  # 逐批次处理
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        feed_dict = {
            model.input_x: x_test[start_id:end_id],
            model.keep_prob: 1.0,
	    model.batch_size:len(x_test[start_id:end_id]),
            model.real_length:get_real_seq_length(x_test[start_id:end_id])
        }
        pred_prob[start_id:end_id] = session.run(model.pred_prob,feed_dict=feed_dict)
        #sorted_pred_prob[start_id:end_id]=[np.argsort(pred_prob[i]) for i in range(start_id,end_id)]
        '''
        for i in range(start_id,end_id):
            if pred_prob[i][int(sorted_pred_prob[i][-1])]>0.85:
                y_pred_cls[i] = sorted_pred_prob[i][-1]
            else:
                y_pred_cls[i] = sorted_pred_prob[i][-2]
        '''
        #print('a')
    for p_y in pred_prob:
        m = []
        for k in range(len(p_y)):
            if p_y[k] >= 0.5:
                m.append(k)
        if len(m) < 1:
            m.append(np.argmax(p_y))
        y_pred_cls.append(m)
    res = []
    res_indexs = []
    for i in range(len(y_test_cls)):
        res.append([contents[i],'#'.join([id_to_cat[_] for _ in y_test_cls[i]]),'#'.join([id_to_cat[_] for _ in y_pred_cls[i]])])
    res = pd.DataFrame(res,columns=['content','real_label','pred_label'])
    res.to_excel('result.xlsx',index=False,encoding='utf-8')
    for i in range(len(y_test_cls)):
        res_indexs.append([contents[i],'#'.join([str(_) for _ in y_test_cls[i]]),'#'.join([str(_) for _ in y_pred_cls[i]])])
    res_indexs = pd.DataFrame(res_indexs, columns=['content', 'real_label', 'pred_label'])
    res_indexs.to_excel('result_index.xlsx', index=False, encoding='utf-8')
        #y_pred_cls[start_id:end_id] = session.run(model.y_pred_cls, feed_dict=feed_dict)
    # TP = defaultdict(int)
    # FP =  defaultdict(int)
    # TN = defaultdict(int)
    # FN = defaultdict(int)
    # for i in range(len(y_test_cls)):
    #     for j in range(len(y_test_cls[i])):
    #         for n in range(len(cat_to_id)):
    #             if y_pred_cls[i][j] == n  and y_test_cls[i][j] == n:
    #                 TP[n] += 1
    #             elif y_pred_cls[i][j] == n  and y_test_cls[i][j] != n:
    #                 FP[n] += 1
    #             elif  y_pred_cls[i][j] != n  and y_test_cls[i][j] == n:
    #                 FN[n] += 1
    #             else:
    #                 TN[n] += 1

    # 评估
    print("Precision, Recall and F1-Score...")
    report_labels = unique_labels(y_test_cls, y_pred_cls)
    p, r, f1, s = metrics.precision_recall_fscore_support(y_test_cls,y_pred_cls,labels=report_labels,average='micro')
    #print(p)
    report = pd.DataFrame([[report_labels[i],p[i],r[i],f1[i],s[i]] for i in range(len(report_labels))],columns=['label','precision','recall','f1','support'])
    #report = pd.DataFrame([p,r,f1,s],columns=['precision','recall','f1','support'])
    report.to_csv('../data/temp/confuse_report.csv',index=False)
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    df = pd.DataFrame(metrics.confusion_matrix(y_test_cls, y_pred_cls))
    df.to_csv('../data/temp/cm.csv',encoding='utf-8')
    '''
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Sheet1')
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            ws.write(i,j,cm[i][j])
    wb.save('../data/temp/cm.xls')
    '''
    print(cm)

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError("""usage: python run_cnn.py [train / test]""")

    print('Configuring CNN model...')
    config = TCNNConfig()

    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, config.vocab_size)
   # w2v = get_word_embedding(w2v_path, vocab_dir, config.embedding_dim)
   # config.w2v = w2v
   # print(w2v)
   # print(config.w2v)
    categories, cat_to_id = read_category(categories_dir)
    id_to_cat = {v:k for k,v in cat_to_id.items()}
    words, word_to_id = read_vocab(vocab_dir)
    #print('loading word embedding...')
    #embeddings = get_embeddings('./datasets/w2v.txt',vocab_dir,word_to_id)
    #embeddings = pickle.load(open('./datasets/embeddings.pkl','rb'))
    #config.embedding_dim = len(embeddings[0])
    config.num_classes = len(cat_to_id)
    config.vocab_size = len(words)
    config.is_w2v = False
    #config.w2v = embeddings
    model = TextCNN(config)

    if sys.argv[1] == 'train':
        train()
    else:
        test()
