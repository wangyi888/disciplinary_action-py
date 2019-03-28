# coding: utf-8

import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import batch_norm
import numpy as np
class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 300  # 词向量维度
    seq_length = 600  # 序列长度
    num_classes = 234  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_sizes = [2,3,4,5,6]  # 卷积核尺寸
    vocab_size = 20000  # 词汇表达小

    fc_dim = 512
    num_units = 128 # rnn隐层神经元个数
    attention_size = 50 # attention隐层元个数

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-4  # 学习率
    decay_rate = 1 # 学习率衰减比率
    decay_steps = 20000 # 衰减步数

    batch_size = 128  # 每批训练大小
    num_epochs = 30  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 10  # 每多少轮存入tensorboard
    is_w2v = False
    def __init__(self,w2v=None):
        self.w2v = w2v
        
class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config):
        self.config = config

        # 三个待输入的数据
        self.input_x = tf.placeholder(tf.int32, [None, self.config.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.config.num_classes], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.batch_size = tf.placeholder(dtype=tf.int32,shape=[])
        self.real_length = tf.placeholder(dtype=tf.int32,shape=[None])
        #self.istraining = tf.placeholder(tf.bool)
        # 词向量映射
        with tf.device('/cpu:0'):

            if self.config.is_w2v:
                embedding = tf.Variable(self.config.w2v,dtype=tf.float32,trainable=True)
            else:
                embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x,name='embedding')
            self.embedding_inputs_expand = tf.expand_dims(self.embedding_inputs, -1)
        #self.rnn_output = self.rnn()
        #self.att_out = self.attention(self.rnn_output,self.config.attention_size)
        #print('a')
       # output = self.attention(self.embedding_inputs,self.config.attention_size)
       # output = tf.reshape(output,[self.batch_size,self.config.seq_length,self.config.embedding_dim])
       # self.output = tf.expand_dims(output,-1)
       # print(output)
        self.cnn()

    # 双向RNN
    def rnn(self,inputs,sequence_length=None):
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units=self.config.num_units)
        cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw,self.keep_prob)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units=self.config.num_units)
        cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw,self.keep_prob)
        initial_state_fw = cell_fw.zero_state(self.batch_size,dtype=tf.float32)
        initial_state_bw = cell_bw.zero_state(self.batch_size,dtype=tf.float32)
        output,state = tf.nn.bidirectional_dynamic_rnn(cell_fw,cell_bw,inputs,sequence_length=sequence_length,initial_state_fw=initial_state_fw,initial_state_bw=initial_state_bw)
        return tf.concat(state,-1)
    def cnn(self):
        """CNN模型"""
        pools = []
        convs = []
        for i,kernel_size in enumerate(self.config.kernel_sizes):
            with tf.variable_scope("cnn"+str(i)):
               # F = tf.Variable(initial_value=tf.truncated_normal([kernel_size,self.config.embedding_dim,1,self.config.num_filters],stddev=np.sqrt(2/kernel_size)),name='w')
                F = tf.get_variable('w',shape=[kernel_size,self.config.embedding_dim,1,self.config.num_filters],initializer=xavier_initializer())
                #F = tf.get_variable('w', shape=[kernel_size, self.config.num_units*2, 1, self.config.num_filters],
                #                    initializer=xavier_initializer())
                B = tf.Variable(tf.constant(0.0,shape=[self.config.num_filters]),name='b')
                # CNN layer
                conv = tf.nn.conv2d(self.embedding_inputs_expand,F,strides=[1,1,1,1],padding='VALID')
                #conv = tf.nn.conv2d(self.output, F, strides=[1, 1, 1, 1], padding='VALID')
               # conv = batch_norm(conv,is_training=self.istraining)
                conv = tf.nn.relu(tf.nn.bias_add(conv,B),name='conv')
               # conv = batch_norm(conv,is_training=self.istraining)
                # chunk-2 pooling
               # conv1 = tf.strided_slice(conv, [0, 0, 0, 0], [self.config.batch_size, int(conv.get_shape()[1].value / 2),
               #                                               conv.get_shape()[2].value, conv.get_shape()[3].value],
               #                          name='conv1')
               # conv2 = tf.strided_slice(conv, [0, int(conv.get_shape()[1].value / 2), 0, 0],
               #                          [self.config.batch_size, conv.get_shape()[1].value,
               #                           conv.get_shape()[2].value, conv.get_shape()[3].value], name='conv2')
               # pool1 = tf.nn.max_pool(conv1, ksize=[1, conv1.get_shape()[1], 1, 1], strides=[1, 1, 1, 1],
               #                        padding='VALID', name='pool1')
               # pool2 = tf.nn.max_pool(conv2, ksize=[1, conv2.get_shape()[1], 1, 1], strides=[1, 1, 1, 1],
               #                        padding='VALID', name='pool2')
               # gmp = tf.concat([pool1, pool2], axis=1)
                # global max pooling layer
                gmp = tf.nn.max_pool(conv, [1, self.config.seq_length-kernel_size+1, 1, 1], [1, 1, 1, 1], padding='VALID', name='gmp')
                gmp = tf.squeeze(gmp,axis=[1,2])
               # convs.append(conv)
                pools.append(gmp)

        #self.pools = tf.concat(pools,3)
        self.pools = tf.concat(pools,-1)
        #real_length = tf.reduce_sum(tf.sign(self.input_x),axis=-1)
        #rnn_final_states = self.rnn(self.embedding_inputs,real_length)
        #fn = tf.concat([self.pools,rnn_final_states],axis=-1)
       # self.convs = tf.squeeze(tf.concat(convs,1),2)#tf.concat(convs,1)
       # self.pool2 = tf.squeeze(self.pools,axis=[1,2])
       # self.rnn_output = self.rnn(self.convs)
        #self.all = tf.concat([self.pool2,self.att_out],axis=1)
        h = tf.reshape(self.pools,[-1,(self.pools.get_shape()[1].value)])
        #h = tf.reshape(fn, [-1, (self.pools.get_shape()[1].value)+rnn_final_states.get_shape()[1].value])
        #h =  tf.reshape(self.rnn_output,[-1,self.rnn_output.get_shape()[1].value*self.rnn_output.get_shape()[2].value])
        with tf.name_scope("score"):
            ds = tf.layers.dense(h,units=self.config.fc_dim,activation=tf.nn.relu,kernel_initializer=xavier_initializer())
            # 全连接层，后面接dropout以及relu激活
            self.w = tf.get_variable('w', shape=[self.config.fc_dim,self.config.num_classes])
            #self.w = tf.get_variable('w',shape=[(self.pools.get_shape()[1].value)*(self.pools.get_shape()[3].value),self.config.num_classes])
            #self.w = tf.get_variable('w', shape=[self.pools.get_shape()[3].value+self.config.num_units*2,self.config.num_classes])
            #self.w = tf.get_variable('w', shape=[self.rnn_output.get_shape()[1].value*self.rnn_output.get_shape()[2].value,self.config.num_classes],initializer=xavier_initializer())
            self.b = tf.Variable(tf.constant(0.0,shape=[self.config.num_classes]),name='b')
            fc = tf.nn.bias_add(tf.matmul(ds,self.w),self.b)

            # 分类器
            self.logits = tf.nn.dropout(fc,self.keep_prob)
            self.pred_prob = tf.nn.sigmoid(self.logits)
            #self.pred_prob = tf.nn.softmax(self.logits)
            #self.y_pred_cls = tf.argmax(tf.nn.sigmoid(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            #cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            global_step = tf.Variable(0)
            self.dynamic_learning_rate = tf.train.exponential_decay(learning_rate=self.config.learning_rate,global_step=global_step,decay_rate=self.config.decay_rate,
                                       decay_steps=self.config.decay_steps)
            # 优化器
            #self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)
            self.optim = tf.train.AdamOptimizer(learning_rate=self.dynamic_learning_rate).minimize(self.loss)
            #self.optim = tf.train.GradientDescentOptimizer(learning_rate=self.dynamic_learning_rate).minimize(self.loss)

        # with tf.name_scope("accuracy"):
        #     # 准确率
        #     correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
        #     self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # 注意力机制
    def attention(self,inputs,attention_size):
        hidden_size = inputs.get_shape()[2].value
        with tf.variable_scope("matrix"):
            W_omega = tf.get_variable('w_omega',shape=[hidden_size,attention_size],dtype=tf.float32,initializer=xavier_initializer())
            b_omega = tf.get_variable('b_omega',shape=[attention_size],dtype=tf.float32,initializer=xavier_initializer())
            u_omega = tf.get_variable('u_omega',shape=[attention_size],dtype=tf.float32,initializer=xavier_initializer())
        with tf.name_scope("attention"):
            u_it = tf.nn.tanh(tf.tensordot(inputs,W_omega,axes=1)+b_omega)
            alpha_it = tf.nn.softmax(tf.tensordot(u_it,u_omega,axes=1))
            att_output = tf.reduce_sum(inputs*tf.expand_dims(alpha_it,-1),axis=1)
        return att_output


