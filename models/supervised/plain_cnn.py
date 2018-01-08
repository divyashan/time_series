import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
import pdb
import sys

sys.path.insert(0, '../../')

# from time_series.tsne_python import tsne
from time_series.parse_dataset.readUcr import UCRDataset
from time_series.models.utils import evaluate_train_embedding



def fix_classes(y_train, y_val, y_test):
  labels = np.unique(y_train)
  label_dict = {labels[i]:i for i in range(len(labels))}
  y_train = np.array([label_dict[x] for x in y_train])
  y_val = np.array([label_dict[x] for x in y_val])
  y_test = np.array([label_dict[x] for x in y_test])
  return y_train, y_val, y_test

def plot_filters(plot_row, n_filters, W):
  f, axarr = plt.subplots(1, n_filters)
  for f_num in range(n_filters):
    plot_f = np.squeeze(W[:,:,:,f_num])
    for n in range(plot_row):
      axarr[f_num].plot(plot_f)
      plt.setp([axarr[f_num].get_xticklabels()], visible=False)
      if not f_num == 0:
          plt.setp([axarr[f_num].get_yticklabels()], visible=False)
  f.subplots_adjust(hspace=0)  #No horizontal space between subplots
  f.subplots_adjust(wspace=0)
  plt.show()

def evaluate_test_embedding(train_embedding, tr_y, test_embedding, test_y):
    n_correct = 0.0
    for sample, correct_label in zip(test_embedding, test_y):
        label = classify_sample(sample, train_embedding, tr_y)
        if label == correct_label:
            n_correct += 1.0
    return n_correct/len(test_y)

def classify_sample(output, train_embedding, tr_y):
    dists = [np.linalg.norm(output-row) for row in train_embedding]
    return tr_y[np.argmin(dists)]


#Check for the input sizes
#assert (N>X_train.shape[1]), 'You are feeding a fat matrix for training, are you sure?'
#assert (Ntest>X_test.shape[1]), 'You are feeding a fat matrix for testing, are you sure?'

#Proclaim the epochs
def test_model(dataset, pool_pctg=.1, layer_size=40, stride_pct=-1):
  tf.reset_default_graph()
  num_filt_1 = 8     #Number of filters in first conv layer
  num_filt_2 = 14     #Number of filters in second conv layer
  num_fc_1 = layer_size      #Number of neurons in hully connected layer
  max_iterations = 20000
  batch_size = 24
  dropout = 1.0      #Dropout rate in the fully connected layer
  plot_row = 5        #How many rows do you want to plot in the visualization
  learning_rate = 2e-5
  input_norm = False   # Do you want z-score input normalization?

  # ucr_dataset = UCRDataset("../ucr_data/" + dataset)
  UCR_DATA_DIR = os.path.expanduser('~/Desktop/datasets/UCR_TS_Archive_2015/')
  ucr_dataset = UCRDataset(UCR_DATA_DIR + dataset)

  X_train = ucr_dataset.Xtrain
  y_train = ucr_dataset.Ytrain
  X_val = ucr_dataset.Xtest[:2]
  y_val = ucr_dataset.Ytest[:2]
  X_test = ucr_dataset.Xtest
  y_test = ucr_dataset.Ytest

  N = X_train.shape[0]
  Ntest = X_test.shape[0]
  D = X_train.shape[1]

  pool_width = pool_pctg*D
  if stride_pct == -1:
    stride_width = 2
  else:
    stride_width = int(np.ceil(max(stride_pct*pool_width, 1)))



  print('We have %s observations with %s dimensions'%(N,D))


  # Organize the classes
  num_classes = len(np.unique(y_train))
  labels = np.unique(y_train)
  #fix_labels = {y_train[i]: i for i in range(len(labels))}
  #y_train = np.array([fix_labels[y_val] for y_val in y_train])
  #y_test = np.array([fix_labels[y_val] for y_val in y_test])
  """
  base = np.min(y_train)  #Check if data is 0-based
  if base != 0:
      y_train -=base
      y_test -= base
  """
  epochs = np.floor(batch_size*max_iterations / N)
  y_train, y_val, y_test = fix_classes(y_train, y_val, y_test)

  # Nodes for the input variables
  x = tf.placeholder("float", shape=[None, D], name = 'Input_data')
  y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
  keep_prob = tf.placeholder("float")
  bn_train = tf.placeholder(tf.bool)          #Boolean value to guide batchnorm

  # Define functions for initializing variables and standard layers
  #For now, this seems superfluous, but in extending the code
  #to many more layers, this will keep our code
  #read-able

  def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name = name)

  def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

  def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, pool_width, 1, 1],
                          strides=[1, stride_width, 1, 1], padding='SAME')

  with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x, [-1,D,1,1])

  initializer = tf.contrib.layers.xavier_initializer()
  """Build the graph"""
  # ewma is the decay for which we update the moving average of the
  # mean and variance in the batch-norm layers
  with tf.name_scope("Conv1") as scope:
    W_conv1 = tf.get_variable("Conv_Layer_1", shape=[5, 1, 1, num_filt_1],initializer=initializer)
    b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
    a_conv1 = conv2d(x_image, W_conv1) + b_conv1

  with tf.name_scope('Batch_norm_conv1') as scope:
      a_conv1 = tf.contrib.layers.batch_norm(a_conv1,is_training=bn_train,updates_collections=None)
      h_conv1 = tf.nn.relu(a_conv1)
      h_conv1 = max_pool_2x2(h_conv1)
  """
  with tf.name_scope("Conv2") as scope:
    W_conv2 = tf.get_variable("Conv_Layer_2", shape=[4, 1, num_filt_1, num_filt_2],initializer=initializer)
    b_conv2 = bias_variable([num_filt_2], 'bias_for_Conv_Layer_2')
    a_conv2 = conv2d(h_conv1, W_conv2) + b_conv2

  with tf.name_scope('Batch_norm_conv2') as scope:
      a_conv2 = tf.contrib.layers.batch_norm(a_conv2,is_training=bn_train,updates_collections=None)
      h_conv2 = tf.nn.relu(a_conv2)
      h_conv2 = max_pool_2x2(h_conv2)
  """

  with tf.name_scope("Fully_Connected1") as scope:
    #W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D*num_filt_2, num_fc_1],initializer=initializer)
    W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[int(np.ceil(D*(1.0/stride_width))*num_filt_1), num_fc_1],initializer=initializer)

    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    #h_conv3_flat = tf.reshape(h_conv2, [-1, D*num_filt_2])
    h_conv3_flat = tf.reshape(h_conv1, [-1, int(np.ceil(D*(1.0/stride_width))*num_filt_1)])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

  with tf.name_scope("Fully_Connected2") as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_classes],initializer=initializer)
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  with tf.name_scope("SoftMax") as scope:
  #    regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) +
  #                  tf.nn.l2_loss(W_conv2) + tf.nn.l2_loss(b_conv2) +
  #                  tf.nn.l2_loss(W_conv3) + tf.nn.l2_loss(b_conv3) +
  #                  tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
  #                  tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2,labels=y_)
      cost = tf.reduce_sum(loss) / batch_size
  #    cost += regularization*regularizers
      loss_summ = tf.summary.scalar("cross entropy_loss", cost)
  with tf.name_scope("train") as scope:
      tvars = tf.trainable_variables()
      #We clip the gradients to prevent explosion
      grads = tf.gradients(cost, tvars)
      optimizer = tf.train.AdamOptimizer(learning_rate)
      gradients = list(zip(grads, tvars))
      train_step = optimizer.apply_gradients(gradients)
      # The following block plots for every trainable variable
      #  - Histogram of the entries of the Tensor
      #  - Histogram of the gradient over the Tensor
      #  - Histogram of the grradient-norm over the Tensor
      numel = tf.constant([[0]])
      for gradient, variable in gradients:
        if isinstance(gradient, ops.IndexedSlices):
          grad_values = gradient.values
        else:
          grad_values = gradient

        numel +=tf.reduce_sum(tf.size(variable))

        h1 = tf.summary.histogram(variable.name, variable)
        h2 = tf.summary.histogram(variable.name + "/gradients", grad_values)
        h3 = tf.summary.histogram(variable.name + "/gradient_norm", clip_ops.global_norm([grad_values]))
  with tf.name_scope("Evaluating_accuracy") as scope:
      correct_prediction = tf.equal(tf.argmax(h_fc2,1), y_)
      accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
      accuracy_summary = tf.summary.scalar("accuracy", accuracy)


  #Define one op to call all summaries
  merged = tf.summary.merge_all()

  # For now, we collect performances in a Numpy array.
  # In future releases, I hope TensorBoard allows for more
  # flexibility in plotting
  perf_collect = np.zeros((3,int(np.floor(max_iterations /100))))
  cost_ma = 0.0
  acc_ma = 0.0

  patience_window = 20
  last_10_val = [0 for i in range(patience_window)]

  with tf.Session() as sess:
    writer = tf.summary.FileWriter("./log_tb", sess.graph)

    sess.run(tf.global_variables_initializer())

    step = 0      # Step is a counter for filling the numpy array perf_collect
    for i in range(max_iterations):
      batch_ind = np.random.choice(N,batch_size,replace=True)

      if i==0:
          # Use this line to check before-and-after test accuracy
          """
          result = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
          acc_test_before = result
          """
      if i%400 == 0:
        #Check training performance

        result = sess.run([cost,accuracy],feed_dict = { x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False})
        perf_collect[1,step] = acc_train = result[1]
        cost_train = result[0]

        #Check validation performance
        result = sess.run([accuracy,cost,merged], feed_dict={ x: X_val, y_: y_val, keep_prob: 1.0, bn_train : False})
        perf_collect[0,step] = acc_val = result[0]
        cost_val = result[1]
        if i == 0: cost_ma = cost_train
        if i == 0: acc_ma = acc_train
        cost_ma = 0.8*cost_ma+0.2*cost_train
        acc_ma = 0.8*acc_ma + 0.2*acc_train

        train_embedding = h_fc1.eval(feed_dict = {x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False })
        #test_embedding = h_fc1.eval(feed_dict = {x: X_test, y_: y_train, keep_prob: 1.0, bn_train : False })
        #gg = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
        #print('Accuracy given NN approach %0.2f' %(100*gg))
        gg = evaluate_train_embedding(train_embedding, y_train)
        last_10_val[(i/200) % patience_window] = gg

        if last_10_val.count(last_10_val[0]) == len(last_10_val) and i > 3000:
          print 'Stopping early!'
          #break

        #Write information to TensorBoard
        #writer.add_summary(result[2], i)
        writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
        print("At %5.0f/%5.0f Cost: train%5.3f val%5.3f(%5.3f) Acc: train%5.3f val%5.3f(%5.3f) " % (i,max_iterations, cost_train,cost_val,cost_ma,acc_train,acc_val,acc_ma))
        step +=1
      sess.run(train_step,feed_dict={x:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout, bn_train : True})
    result = sess.run([accuracy,numel], feed_dict={ x: X_test, y_: y_test, keep_prob: 1.0, bn_train : False})
    acc_test = result[0]
    print('The network has %s trainable parameters'%(result[1]))
    filter1 = sess.run(W_conv1)
    #plot_filters(1, num_filt_1, filter1)

    train_embedding = h_fc1.eval(feed_dict = {x: X_train, y_: y_train, keep_prob: 1.0, bn_train : False })
    test_embedding = h_fc1.eval(feed_dict = {x: X_test, y_: y_train, keep_prob: 1.0, bn_train : False })
    gg = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
    print('Accuracy given NN approach %0.2f' %(100*gg))

  """Additional plots"""
  return gg
  #plt.show()
  # We can now open TensorBoard. Run the following line from your terminal
  # tensorboard --logdir=./log_tb

if __name__ == '__main__':
  test_model(sys.argv[1], .1, 40)
