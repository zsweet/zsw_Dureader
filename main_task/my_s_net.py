import tensorflow as tf
from layers.basic_rnn import rnn
import tensorflow.contrib as tc

class AttnCell(tc.rnn.LSTMCell):
    """
    Implements the Match-LSTM attention cell
    """
    def __init__(self, num_units, context_to_attend):
        super(AttnCell, self).__init__(num_units, state_is_tuple=True)
        self.context_to_attend = context_to_attend
        ### Hq * Wq =>  Q*l + l*l =>  Q*l
        self.fc_context = tc.layers.fully_connected(self.context_to_attend,num_outputs=self._num_units,activation_fn=None)

    def __call__(self, inputs, state, scope=None):
        (c_prev, h_prev) = state
        with tf.variable_scope(scope or type(self).__name__):
            ref_vector = tf.concat([inputs, h_prev], -1)  ###LSTM输入和隐状态合并
            zswtmp = tc.layers.fully_connected(ref_vector,num_outputs=self._num_units,activation_fn=None) ### 300 * P
            G = tf.tanh(self.fc_context + tf.expand_dims(zswtmp, 1))
            logits = tc.layers.fully_connected(G, num_outputs=1, activation_fn=None)  ### Q*(l) * (l)* 1 => Q*1
            scores = tf.nn.softmax(logits, 1)
            attended_context = tf.reduce_sum(self.context_to_attend * scores, axis=1)
            new_inputs = tf.concat([inputs, attended_context, inputs - attended_context, inputs * attended_context],-1)
            return super(AttnCell, self).__call__(new_inputs, state, scope)



class S_netModel():
    def __init__(self,vocab,p_length,q_emb,q_length,hiden_size,dropout_keep_prob = 1):  ####!!!!!!注意修改！！！！！！！！
        # the vocab
        self.vocab = vocab

        #encode context
        self._encode()

    def _setup_placeholders(self):
        """
        Placeholders
        """
        self.p = tf.placeholder(tf.int32, [None, None])
        self.q = tf.placeholder(tf.int32, [None, None])
        self.p_length = tf.placeholder(tf.int32, [None])
        self.q_length = tf.placeholder(tf.int32, [None])
        self.start_label = tf.placeholder(tf.int32, [None])
        self.end_label = tf.placeholder(tf.int32, [None])
        self.dropout_keep_prob = tf.placeholder(tf.float32)


    def _embed(self):
        """
        The embedding layer, question and passage share embeddings
        """
        with tf.device('/cpu:0'), tf.variable_scope('word_embedding'):
            self.word_embeddings = tf.get_variable(
                'word_embeddings',
                shape=(self.vocab.size(), self.vocab.embed_dim),
                initializer=tf.constant_initializer(self.vocab.embeddings),
                trainable=True
            )
            self.p_emb = tf.nn.embedding_lookup(self.word_embeddings, self.p)
            self.q_emb = tf.nn.embedding_lookup(self.word_embeddings, self.q)


    def _encode(self):
        """
        Employs two Bi-LSTMs to encode passage and question separately
        """
        with tf.variable_scope('passage_encoding'):
            self.sep_p_encodes, _ = rnn('bi-gru', self.p_emb, self.p_length, self.hidden_size)   ###300维
        with tf.variable_scope('question_encoding'):
            self.sep_q_encodes, _ = rnn('bi-gru', self.q_emb, self.q_length, self.hidden_size)  ##???????????
        if self.use_dropout:
            self.sep_p_encodes = tf.nn.dropout(self.sep_p_encodes, self.dropout_keep_prob)  ###T*300
            self.sep_q_encodes = tf.nn.dropout(self.sep_q_encodes, self.dropout_keep_prob)  ###J*300
        print ('paragram',self.sep_p_encodes,'qestion',self.sep_q_encodes)

    class MatchLSTMLayer(object):
        """
        Implements the Match-LSTM layer, which attend to the question dynamically in a LSTM fashion.
        """

        def __init__(self, hidden_size):
            self.hidden_size = hidden_size

        def match(self, passage_encodes, question_encodes, p_length,
                  q_length):  ###passage_encodes, question_encodes 300维
            """
            Match the passage_encodes with question_encodes using Match-LSTM algorithm
            """
            with tf.variable_scope('match_lstm'):
                cell_fw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
                cell_bw = MatchLSTMAttnCell(self.hidden_size, question_encodes)
                outputs, state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw,
                                                                 inputs=passage_encodes,
                                                                 sequence_length=p_length,
                                                                 dtype=tf.float32)
                match_outputs = tf.concat(outputs, 2)
                state_fw, state_bw = state
                c_fw, h_fw = state_fw
                c_bw, h_bw = state_bw
                match_state = tf.concat([h_fw, h_bw], 1)
            return match_outputs, match_state

if __name__  == '__main__':
    p_emb=''
    p_length=''
    q_emb=''
    q_length=''
    hiden_size=''
    dropout_keep_prob = 1
    s_net = S_netModel(p_emb,p_length,q_emb,q_length,hiden_size,dropout_keep_prob = 1)