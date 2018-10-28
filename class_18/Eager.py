import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.enable_eager_execution()
tfe = tf.contrib.eager

class Model(tf.keras.Model):
    def __init__(self):
        super( Model, self ).__init__()
        self.conv1 = tf.layers.Conv2D(64, [3,3])
        self.pool1 = tf.layers.MaxPooling2D( [2,2], [1,1] )

        self.conv2 = tf.layers.Conv2D(32, [2, 2])
        self.pool2 = tf.layers.MaxPooling2D([2, 2], [1,1])

        self.fc1 = tf.layers.Dense(64)
        self.fc2 = tf.layers.Dense(10)

    def frwrd_pass(self, inp):
        inp = tf.reshape( inp, [-1,28,28,1] )
        conv1 = self.conv1(inp)
        pool1 = self.pool1(conv1)
        pool1 = tf.nn.relu( pool1 )

        conv2 = self.conv2(pool1)
        pool2 = self.pool2( conv2 )
        pool2 = tf.nn.relu( pool2 )

        flatten = tf.layers.flatten(pool2)
        fc1 = self.fc1( flatten )
        fc1 = tf.nn.relu( fc1 )

        fc2 = self.fc2(fc1)
        return fc2

model = Model()

def loss(pred, desired):
    cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=desired)
    cost = tf.reduce_mean(cost)
    return cost

def train(X_train, Y_train, num_epoch):
    opt = tf.train.AdamOptimizer(learning_rate=0.01)

    for epoch in range(num_epoch):

        with tf.GradientTape() as tape:
            predicted = model.frwrd_pass(X_train)
            curr_loss = loss(predicted, Y_train)

        grads = tape.gradient( curr_loss, model.variables )
        opt.apply_gradients(zip( grads, model.variables ),
                            global_step=tf.train.get_or_create_global_step())

        print ("Loss at epoch " + str(epoch) + " " + str( loss( model.frwrd_pass(X_train), Y_train  ) ) )
        print ("\n -------- \n")

mnist = input_data.read_data_sets("data", one_hot=True)

batch = mnist.train.next_batch(1000)

X_train = batch[0]
Y_train = batch[1]

train(X_train/255.0, Y_train, 15)
