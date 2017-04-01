import tensorflow as tf
import tensorflow.contrib as con
from loader import VAR_LABEL, FRAME_COUNT, DELAY

HIDDEN_SIZE = 8
NUM_LAYERS = 4
max_grad_norm = 2

HIDDEN_SIZE_2 = 8
NUM_LAYERS_2 = 2



def inference(data, prob, BATCH_SIZE, delay):
    with tf.variable_scope("RNN"):
        lstm_cell = con.rnn.LSTMCell(HIDDEN_SIZE)
        lstm_cell_drop = con.rnn.DropoutWrapper(lstm_cell, output_keep_prob=prob)
        cell = con.rnn.MultiRNNCell([lstm_cell_drop for _ in range(NUM_LAYERS)])
        _initial_state = cell.zero_state(BATCH_SIZE, tf.float32)

        data = tf.nn.dropout(data, prob)
        # inputs = tf.unstack(data, num=FRAME_COUNT, axis=1)
        # outputs, state = tf.nn.dynamic_rnn(cell, data, initial_state=cell.zero_state(BATCH_SIZE, tf.float32))

        outputs = []
        state = _initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(data.shape[1]):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(data[:, time_step, :], state)
                outputs.append(cell_output)


    # with tf.variable_scope("Hidden_1"):
    #     delay = tf.reshape(delay, [-1, DELAY * 2])
    #     W_hidden1 = tf.Variable(tf.truncated_normal(shape=[DELAY * 2, 8], stddev=0.01),
    #                             dtype=tf.float32,
    #                             name='W_hidden1')
    #     b_hidden1 = tf.Variable(tf.constant(0.0, shape=[8]),
    #                             dtype=tf.float32,
    #                             name='b_hidden1')
    #     h_hidden1 = tf.nn.relu(tf.matmul(delay, W_hidden1) + b_hidden1)
    #     weight_decay = tf.multiply(tf.nn.l2_loss(W_hidden1), 5e-4, name='weight_loss')
    #     tf.add_to_collection('losses', weight_decay)

    with tf.variable_scope("RNN2"):
        temp_batch_size = int(delay.shape[0] * delay.shape[1])
        lstm_cell_2 = con.rnn.LSTMCell(HIDDEN_SIZE_2)
        lstm_cell_drop_2 = con.rnn.DropoutWrapper(lstm_cell_2, output_keep_prob=prob)
        cell_2 = con.rnn.MultiRNNCell([lstm_cell_drop_2 for _ in range(NUM_LAYERS_2)])
        _initial_state_2 = cell_2.zero_state(temp_batch_size, tf.float32)

        delay = tf.reshape(delay, [temp_batch_size, DELAY, 2])
        delay = tf.nn.dropout(delay, prob)

        state_2 = _initial_state_2
        with tf.variable_scope("RNN2"):
            for time_step in range(DELAY):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output_2, state_2) = cell_2(delay[:, DELAY - 1 - time_step, :], state_2)


    with tf.variable_scope("softmax"):
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])
        softmax_w = tf.Variable(tf.truncated_normal(shape=[HIDDEN_SIZE, VAR_LABEL + 1], stddev=0.01),
                                dtype=tf.float32,
                                name='softmax_w')
        softmax_b = tf.Variable(tf.constant(0.0, shape=[VAR_LABEL + 1]),
                                dtype=tf.float32,
                                name='softmax_b')
        delay_w = tf.Variable(tf.truncated_normal(shape=[HIDDEN_SIZE_2, VAR_LABEL + 1], stddev=0.01),
                              dtype=tf.float32,
                              name='delay_w')
        logits = tf.add(tf.add(tf.matmul(output, softmax_w), tf.matmul(cell_output_2, delay_w)), softmax_b, name='logits')


    with tf.variable_scope("output"):
        argmax = tf.nn.softmax(logits, name='argmax')
        state_0_c = tf.multiply(state[0].c, 1, name='state_0_c')
        state_0_h = tf.multiply(state[0].h, 1, name='state_0_h')
        state_1_c = tf.multiply(state[1].c, 1, name='state_1_c')
        state_1_h = tf.multiply(state[1].h, 1, name='state_1_h')
        state_2_c = tf.multiply(state[2].c, 1, name='state_2_c')
        state_2_h = tf.multiply(state[2].h, 1, name='state_2_h')
        state_3_c = tf.multiply(state[3].c, 1, name='state_3_c')
        state_3_h = tf.multiply(state[3].h, 1, name='state_3_h')

    return logits, _initial_state, state


def total_loss(logits, labels, batch_size):
    loss = con.legacy_seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(labels, [-1])],
        [tf.ones([batch_size * FRAME_COUNT])])
    loss_ave = tf.reduce_sum(loss) / batch_size
    # return loss_ave
    tf.add_to_collection('losses', loss_ave)
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def myTrain(loss, learning_rate, batch):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=batch)
    return train_op


