import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

num_epochs  = 20
learning_rate = 0.01

num_input = 784
num_classes = 10

layer_1_nodes = 128
layer_2_nodes = 64

display_step = 5


with tf.variable_scope('input'):
    X = tf.placeholder( dtype=tf.float32, shape=[None, num_input] )

with tf.variable_scope('layer_1'):
    weights_1 = tf.Variable( tf.random_normal( shape= [num_input, layer_1_nodes] ), name = 'weights_1' )
    biases_1 = tf.Variable( tf.zeros( shape= [layer_1_nodes] ), name = 'biases_1' )
    layer_1_output = tf.matmul( X, weights_1 ) + biases_1
    layer_1_output = tf.nn.relu( layer_1_output )

with tf.variable_scope('layer_2'):
    weights_2 = tf.Variable( tf.random_normal( shape= [layer_1_nodes, layer_2_nodes] ), name = 'weights_2' )
    biases_2 = tf.Variable( tf.zeros( shape= [layer_2_nodes] ), name = 'biases_2' )
    layer_2_output = tf.matmul( layer_1_output, weights_2 ) + biases_2
    layer_2_output = tf.nn.relu( layer_2_output )

with tf.variable_scope('output'):
    weights_out = tf.Variable( tf.random_normal( shape= [layer_2_nodes, num_classes] ), name = 'weights_3' )
    biases_out = tf.Variable( tf.zeros( shape= [num_classes] ), name = 'biases_3' )
    output = tf.matmul( layer_2_output, weights_out ) + biases_out


with tf.variable_scope('cost'):
    Y = tf.placeholder( shape=[None, num_classes], dtype=tf.float32 )
    cost = tf.nn.softmax_cross_entropy_with_logits( logits=output, labels=Y )
    cost = tf.reduce_mean( cost )

mnist = input_data.read_data_sets( 'data', one_hot=True )

with tf.variable_scope('train'):
    opt = tf.train.AdamOptimizer( learning_rate=learning_rate ).minimize(cost)

train_writer = tf.summary.FileWriter('./logs/training')
train_writer.add_graph( tf.get_default_graph() )

with tf.variable_scope('logging'):
    summary = tf.summary.scalar( 'cost', cost )

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    last_ckpt = tf.train.latest_checkpoint('./weights')
    saver.restore( sess, last_ckpt )

    for epoch in range(num_epochs):

        for _ in range(100):

            batch = mnist.train.next_batch(32)

            sess.run( opt, feed_dict= {
                X: batch[0],
                Y: batch[1]
            } )

        batch = mnist.train.next_batch(256)
        training_loss, train_summary = sess.run( [cost, summary] , feed_dict={
            X: batch[0],
            Y: batch[1]
        } )

        train_writer.add_summary( train_summary, epoch )


        test_loss = 0.0
        for _ in range(50):
            batch = mnist.test.next_batch(32)
            test_loss += sess.run( cost, feed_dict={
                X: batch[0],
                Y: batch[1]
            } )

        test_loss = test_loss/50

        if ( epoch + 1 ) % display_step == 0:
            print ( 'Training Loss is ' + str(training_loss) + ' Testing loss is ' + str(test_loss) + '\n')
            saver.save(sess, './weights/epoch_' + str(epoch + 1) + '.ckpt')
            print (" Model Saved \n ")
            print (" ----------- \n")
















