import tensorflow as tf
from random import randint, seed
seed(42)

current_x = tf.placeholder(tf.float32)

x = tf.Variable(2.1, name='x', dtype=tf.float32)
log_x = tf.log(x)
result = current_x * tf.square(log_x)

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(result)

init = tf.initialize_all_variables()

data = {
    "x_1": 1.0,
    "x_2": -1.0,
    "x_3": 1.0
}

def optimize():
  with tf.Session() as session:
    session.run(init)
    for step in range(10):
      feed_dict = get_feed_dict_for(randint(0,2))
      session.run(train, feed_dict=feed_dict)
      loss_for_current_data = get_loss_for(session, feed_dict)
      total_loss = get_total_loss(session)
      print(step, "picked", feed_dict[current_x], "current loss", loss_for_current_data, "loss_total", total_loss)
        
def get_feed_dict_for(i):
    data_key = data.keys()[i]
    return { current_x: data[data_key] }

def get_loss_for(session, feed_dict):
    return session.run(result, feed_dict=feed_dict)

def get_total_loss(session):
    total_loss = 0
    for i in range(3):
      feed_dict = get_feed_dict_for(i)
      total_loss += get_loss_for(session, feed_dict)
    return total_loss

optimize()