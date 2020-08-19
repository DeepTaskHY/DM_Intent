import tensorflow as tf


class Model():
    def __init__(self, rnn_size, vocabulary_size, sequence_len, embedding_size, attention_size, learning_rate, l2_reg_lambda, n_label, index_front, index_back, rnn_cell):
        self.rnn_size = rnn_size
        self.vocabulary_size = vocabulary_size
        self.sequence_len = sequence_len
        self.embedding_size = embedding_size
        self.attention_size = attention_size
        self.learning_rate = learning_rate
        self.l2_reg_lambda = l2_reg_lambda
        self.n_label = n_label
        self.index_front = index_front
        self.index_back = index_back
        self.rnn_cell = rnn_cell
        self.build_model()

    def build_model(self):
        self.inputs = tf.placeholder(tf.int32, [None, self.sequence_len], name='inputs')
        self.inputs_length = tf.placeholder(tf.int32, [None], name='inputs_length')

        self.targets = tf.placeholder(tf.int32, [None], name='targets')
        self.labels = tf.one_hot(self.targets, self.n_label)
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        with tf.variable_scope('RNN'):
            if self.rnn_cell == 'GRU':
                fw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size)
                bw_cell = tf.nn.rnn_cell.GRUCell(num_units=self.rnn_size)
            elif self.rnn_cell == 'LNLSTM':
                fw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.rnn_size, dropout_keep_prob=self.dropout_keep_prob)
                bw_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units=self.rnn_size, dropout_keep_prob=self.dropout_keep_prob)
            else:
                fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)
                bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_size)
            if self.rnn_cell == 'LNLSTM':
                pass
            else:
                fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
                bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)

            embedding_front = tf.get_variable('embedding_front', [self.index_front, self.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer(uniform=False))
            self.embedding_back = tf.get_variable('embedding_back', [self.index_back, self.embedding_size],
                                                  initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                  trainable=False)
            self.embedding = tf.concat([embedding_front, self.embedding_back], 0)
            inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.inputs)
            ((fw_outputs, bw_outputs), (fw_state, bw_state)) = (tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell, cell_bw=bw_cell, inputs=inputs_embedded, sequence_length=self.inputs_length, dtype=tf.float32))
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            outputs = tf.contrib.layers.fully_connected(outputs, self.rnn_size, activation_fn=tf.tanh)

        with tf.variable_scope('attention'):
            w_attention = tf.Variable(tf.random_normal([int(self.rnn_size), self.attention_size], stddev=0.1))
            b_attention = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))
            u_attention = tf.Variable(tf.random_normal([self.attention_size], stddev=0.1))

            v = tf.tanh(tf.tensordot(outputs, w_attention, axes=1) + b_attention)
            vu = tf.tensordot(v, u_attention, axes=1)
            alphas = tf.nn.softmax(vu)

            last = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
            last = tf.nn.dropout(last, self.dropout_keep_prob)

        with tf.variable_scope('score'):
            W = tf.get_variable("W", shape=[self.rnn_size, self.n_label], initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.n_label]), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(last, W, b)
            self.probability = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.variable_scope('optimize'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels)
            self.loss = tf.reduce_mean(cross_entropy) + self.l2_reg_lambda * l2_loss
            self.opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.variable_scope('accuracy'):
            correct_predict = tf.equal(tf.argmax(self.labels, 1), self.predict)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))
        self.saver = tf.train.Saver(tf.global_variables())
