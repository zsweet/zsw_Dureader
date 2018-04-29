import tensorflow as tf

INF = 1e30



def summ(memory, hidden, mask, keep_prob=1.0, is_train=None, scope="summ"):
    with tf.variable_scope(scope):
        d_memory = memory
        s0 = tf.nn.tanh(dense(d_memory, hidden, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * memory, axis=1)
        return res



def softmax_mask( val, mask):
    return -INF * (1 - tf.cast(mask, tf.float32)) + val



def dense(inputs, hidden, use_bias=True, scope="dense"):
    with tf.variable_scope(scope):
        shape = tf.shape(inputs)
        dim = inputs.get_shape().as_list()[-1]
        out_shape = [shape[idx] for idx in range(
            len(inputs.get_shape().as_list()) - 1)] + [hidden]
        flat_inputs = tf.reshape(inputs, [-1, dim])
        W = tf.get_variable("W", [dim, hidden])
        res = tf.matmul(flat_inputs, W)
        if use_bias:
            b = tf.get_variable(
                "b", [hidden], initializer=tf.constant_initializer(0.))
            res = tf.nn.bias_add(res, b)
        res = tf.reshape(res, out_shape)
        return res

def pointer(inputs, state, hidden, mask, scope="pointer"):
    with tf.variable_scope(scope):
        u = tf.concat([tf.tile(tf.expand_dims(state, axis=1), [
            1, tf.shape(inputs)[1], 1]), inputs], axis=2)
        s0 = tf.nn.tanh(dense(u, hidden, use_bias=False, scope="s0"))
        s = dense(s0, 1, use_bias=False, scope="s")
        s1 = softmax_mask(tf.squeeze(s, [2]), mask)
        a = tf.expand_dims(tf.nn.softmax(s1), axis=2)
        res = tf.reduce_sum(a * inputs, axis=1)
        return res, s1

class pr_attention():
    def __init__(self, batch, hidden, keep_prob=1.0, is_train=None, scope="pr_attention"):
        self.batch = batch
        self.scope = scope
        self.keep_prob = keep_prob
        self.is_train = is_train
        self.dropout_mask = tf.ones([batch, hidden], dtype=tf.float32)

    def __call__(self, init, match, d, mask,reuse_flag):
        with tf.variable_scope(self.scope, reuse = reuse_flag):

            inp, logits1 = pointer(match, init * self.dropout_mask, d, mask, "pr_pointer")
            return inp

class attention():
    def __init__(self,hidden):
        self.hidden = hidden

    def __call__(self, inputs, memory, scope="dot_attention"):
        with tf.variable_scope(scope):
            with tf.variable_scope("attention"):
                inputs_ = self.dense(inputs, self.hidden, use_bias=False, scope="inputs")
                memory_ = self.dense(memory, self.hidden, use_bias=False, scope="memory")
                outputs = tf.matmul(inputs_, tf.transpose( memory_, [0, 2, 1]))
                logits = tf.nn.softmax(outputs)
                outputs = tf.matmul(logits, memory)
                res = tf.concat([inputs, outputs], axis=2)

                with tf.variable_scope("gate"):
                    dim = res.get_shape().as_list()[-1]
                    gate = tf.nn.sigmoid(dense(res, dim, use_bias=False))
                return res * gate


    def dense(self, inputs, hidden, use_bias=True, scope="dense"):
        with tf.variable_scope(scope):
            shape = tf.shape(inputs)
            dim = inputs.get_shape().as_list()[-1]
            out_shape = [shape[idx] for idx in range(
                len(inputs.get_shape().as_list()) - 1)] + [hidden]
            flat_inputs = tf.reshape(inputs, [-1, dim])
            W = tf.get_variable("W", [dim, hidden])
            res = tf.matmul(flat_inputs, W)
            if use_bias:
                b = tf.get_variable(
                    "b", [hidden], initializer=tf.constant_initializer(0.))
                res = tf.nn.bias_add(res, b)
            res = tf.reshape(res, out_shape)
            return res