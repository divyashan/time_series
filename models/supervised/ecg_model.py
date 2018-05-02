import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops

import pdb
import sys
import seaborn
import pandas as pd

seaborn.set_style("whitegrid")
sys.path.insert(0, '../')
from time_series.parse_dataset.readECG import loadECG


from time_series.models.utils import evaluate_test_embedding, standardize_ts_lengths, UCR_DATASETS, MV_DATASETS, evaluate_KNN, compute_pairwise_distances, compute_dtw_pairwise_distances
from time_series.models.utils import compute_distances_to_points, compute_dtw_distances_to_points
from scipy.stats import spearmanr
from lifelines import CoxPHFitter


from time_series.models.ecg_utils import *

POOL_PCTG = .05
STRIDE_WIDTH = 10
PRINT = True
FILTER_SIZE = 10
PADDED_LENGTH = 260

"""Hyperparameters"""
num_filt_1 = 2    #Number of filters in first conv layer
num_fc_1 = 2       #Number of neurons in fully connected layer
num_fc_0 = 2
max_iterations = 4000
batch_size = 128
dropout = 1      #Dropout rate in the fully connected layer
learning_rate = 2e-4


def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name = name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, pool_width):
  return tf.nn.max_pool(x, ksize=[1, pool_width, 1, 1],
                        strides=[1, STRIDE_WIDTH, 1, 1], padding='SAME')

def test_model(dataset, pool_pctg, layer_size_1):

  X_train, y_train, X_test, y_test = loadECG()

  X_train = np.swapaxes(X_train, 1, 2)
  X_test = np.swapaxes(X_test, 1, 2)

  n = max([np.max([v.shape[0] for v in X_train]), np.max([v.shape[0] for v in X_test])])
  if n % STRIDE_WIDTH != 0:
    n = n + (STRIDE_WIDTH - (n % STRIDE_WIDTH))

  X_train = standardize_ts_lengths(X_train, n)
  X_test = standardize_ts_lengths(X_test, n)

  X_train = np.swapaxes(X_train, 1, 2)
  X_test = np.swapaxes(X_test, 1, 2)
  N = X_train.shape[0]
  Ntest = X_test.shape[0]
  D = X_train.shape[1]
  D_ts = X_train.shape[2]

  print "X shape: ", X_train.shape[0] ,X_train.shape[1], X_train.shape[2]
  X_val = X_test[:2]
  y_val = y_test[:2]
  X_test = X_test[2:]
  y_test = y_test[2:]

  num_classes = len(np.unique(y_train))
  num_fc_1 = layer_size_1
  epochs = np.floor(batch_size*max_iterations / N)
  pool_width = max(int(POOL_PCTG*D),2)
  print('Train with approximately %d epochs' %(epochs))

  x_tensor = tf.placeholder("float", shape=[None, D, D_ts], name = 'Input_data')
  y_ = tf.placeholder(tf.int64, shape=[None], name = 'Ground_truth')
  keep_prob = tf.placeholder("float")

  with tf.name_scope("Reshaping_data") as scope:
    x_image = tf.reshape(x_tensor, [-1,D,D_ts,1])

    x_image_fc_input = tf.reshape
    initializer = tf.contrib.layers.xavier_initializer()
    """Build the graph"""
    # ewma is the decay for which we update the moving average of the
    # mean and variance in the batch-norm layers
  with tf.name_scope("Conv1") as scope:
    W_conv1 = tf.get_variable("Conv_Layer_1", shape=[1, FILTER_SIZE, 1, num_filt_1],initializer=initializer)
    b_conv1 = bias_variable([num_filt_1], 'bias_for_Conv_Layer_1')
    a_conv1 = conv2d(x_image, W_conv1) + b_conv1

    h_relu = tf.nn.relu(a_conv1)
    h_conv1 = max_pool_2x2(h_relu, pool_width)


  with tf.name_scope("Globally_Informed") as scope:
   W_fc0 = tf.get_variable("Fully_Connected_0", shape=[PADDED_LENGTH, num_fc_0], initializer=initializer)
   b_fc0 = bias_variable([num_fc_0], 'bias_for_Fully_Connected_Layer_0')
   h_fc0 = tf.nn.relu(tf.matmul(tf.contrib.layers.flatten(x_image), W_fc0) + b_fc0)

  # Output of convolutional layer and the fully informed one go into 
  with tf.name_scope("Fully_Connected1") as scope:
    # Code for network without fully-connected inputs
    #h_conv3_flat = tf.contrib.layers.flatten(h_conv1)
    #W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D*num_filt_1*D_ts*(1./STRIDE_WIDTH), num_fc_1],initializer=initializer)

    h_conv3_flat = tf.concat([tf.contrib.layers.flatten(h_conv1), h_fc0], 1)
    W_fc1 = tf.get_variable("Fully_Connected_layer_1", shape=[D*num_filt_1*D_ts*(1./STRIDE_WIDTH) + num_fc_0, num_fc_1],initializer=initializer)
    
    b_fc1 = bias_variable([num_fc_1], 'bias_for_Fully_Connected_Layer_1')
    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)


  with tf.name_scope("Fully_Connected2") as scope:
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    W_fc2 = tf.get_variable("W_fc2", shape=[num_fc_1, num_classes],initializer=initializer)
    b_fc2 = tf.Variable(tf.constant(0.1, shape=[num_classes]),name = 'b_fc2')
    h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

  with tf.name_scope("SoftMax") as scope:
      regularization = .001
      regularizers = (tf.nn.l2_loss(W_conv1) + tf.nn.l2_loss(b_conv1) + tf.nn.l2_loss(W_fc0) + tf.nn.l2_loss(b_fc0)
                    + tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))
      loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h_fc2,labels=y_)

      cost = tf.reduce_sum(loss) / batch_size
      cost += regularization*regularizers
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


  test_normal_ids =  np.array([1071, 1072, 1073, 1075, 1076,  108, 1086, 1088, 1089,  109, 1090, 1091, 1092, 1093, 1094, 1095,   11,  110, 1104, 1105, 1106, 1107, 1109,  111, 1110, 1111, 1112, 1113, 1114, 1115, 1116, 1117, 1118, 1119,  112, 1120, 1121, 1123, 1124, 1126, 1127, 1128, 1129,  113, 1130, 1131, 1133, 1134, 1135, 1138, 1139,  114, 1140, 1141, 1142, 1143, 1146, 1147, 1148, 1149,  115,  116,  117, 1171, 1176, 1179, 118, 1182, 1188,  119,   12,  120,  121,  123, 1232,  124,  125, 126,  127,  128,  129,   13,  130,  131,  132,  133, 1332,  134, 135,  136,  137,  138,  139,  140, 1417,  144, 1463,   15, 1547, 1554,   16, 1616, 1644, 1687,   17, 1723, 1725, 1727, 1731, 1733, 1746,   18, 1804,   19, 1902, 1936, 1992,   20, 2004,   21, 2152, 22, 2234, 2267, 2290,   23, 2390,   24,   25, 2507, 2555, 2556, 2592,   26,   27, 2705, 2779,   28, 2812, 2831, 2832, 2833, 2841, 2844, 2865,   29, 2965, 2966, 2987,    3,   30, 3010, 3014, 3016, 3018, 3040, 3054, 3063, 3077,   31, 3181,   32, 3212, 3217, 3224, 3239, 3249, 3255, 3266, 3268,   33,   34, 3445, 3495, 3522, 3561, 3566, 3575,   36, 3621, 3671, 3679,  368, 3680,  369,   37,  370, 3704,  371, 3717,  372, 3722, 3723, 3727,  373,  374, 3748,  375, 3751,  376,  377, 3779, 3787,  379,   38, 3801,  381,  382, 3861, 39, 3938, 3950, 3957,    4,   40, 4002, 4010, 4018, 4073, 4096, 4132, 4169, 4171, 4174, 4193,   42, 4236, 4240, 4245, 4248, 4257, 4263,   43,  437,  439, 4391,   44, 4407,  441, 4419,  442, 4442, 4450, 4468,   45, 4517, 4523, 4585,   46,   47, 4741, 4747, 4755, 4756, 4766, 4840, 4841, 4869, 4882,   49, 4917,  494,  495, 4954, 497, 4975, 4976,  498, 4980, 4982,  499, 4992,    5,   50,  500, 5004,  502,  503, 5065, 5072, 5083, 5088, 5098,   51, 5106, 5120, 5123, 5144, 5157, 5171])
  test_death_ids = np.array([5616, 5687, 5908, 5964, 6160, 6311, 760, 832, 854, 961])
  merged = tf.summary.merge_all()

  saver = tf.train.Saver()
  with tf.Session() as sess:
    #saver.restore(sess, "./models/model.ckpt")
    writer = tf.summary.FileWriter("./log_tb", sess.graph)

    sess.run(tf.global_variables_initializer())

    step = 0      # Step is a counter for filling the numpy array perf_collect

    i = 0
    def model1deathpred(pid, train_embedding, threshold=None):
      aa = get_all_adjacent_beats(pid)
      aa = standardize_ts_lengths(aa, PADDED_LENGTH)
      aa = np.swapaxes(aa, 1, 2)
      embedded_signal = h_fc1.eval(feed_dict = {x_tensor: aa, y_: y_train, keep_prob: 1.0})
      return percentage_death(train_embedding, y_train, embedded_signal, threshold)
    stop = False
    while stop == False:

      batch_ind = np.random.choice(N,batch_size,replace=False)
      #batch_ind = np.arange(N)
      #batch_ind = batch_ind[(i*batch_size)%N:((i+1)*batch_size)%N]
      gg = h_relu.eval(feed_dict={x_tensor:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout})
      sess.run(train_step,feed_dict={x_tensor:X_train[batch_ind], y_: y_train[batch_ind], keep_prob: dropout})

      if i==0:
          # Use this line to check before-and-after test accuracy
          result = sess.run(accuracy, feed_dict={ x_tensor: X_test, y_: y_test, keep_prob: 1.0})
          acc_test_before = result

      if i%750 == 0 and i != 0:
        #Check training performance

        result = sess.run([cost,accuracy],feed_dict = { x_tensor: X_train, y_: y_train, keep_prob: 1.0})
        acc_train = result[1]
        cost_train = result[0]

    


        if i%750 == 0 and i != 0:
          #print "Running odds ratio calculation"
          #train_embedding = h_fc1.eval(feed_dict = {x_tensor: X_train, y_: y_train, keep_prob: 1.0})
          #get_upper_quartile_odds_ratio(test_death_ids, test_normal_ids, x, y_, y_train, h_fc1, keep_prob, train_embedding)
          train_embedding = h_fc1.eval(feed_dict = {x_tensor: X_train, y_: y_train, keep_prob: 1.0})
          test_embedding = h_fc1.eval(feed_dict = {x_tensor: X_test, y_: y_train, keep_prob: 1.0})
          test_acc = evaluate_test_embedding(train_embedding, y_train, test_embedding, y_test)
          print "test acc: ", test_acc 
          if test_acc > .67:
            pdb.set_trace()

       # writer.add_summary(result[1], i)
        writer.flush()  #Don't forget this command! It makes sure Python writes the summaries to the log-file
        if PRINT == True:
          print("At %5.0f/%5.0f Cost: train%5.3f Acc: train%5.3f " % (i,max_iterations, cost_train, acc_train))
        step +=1
      
      i += 1
    

    result = sess.run([accuracy,numel], feed_dict={ x_tensor: X_test, y_: y_test, keep_prob: 1.0})
    acc_test = result[0]
    print('The network has %s trainable parameters'%(result[1]))

    train_embedding = h_fc1.eval(feed_dict = {x_tensor: X_train, y_: y_train, keep_prob: 1.0})
    test_embedding = h_fc1.eval(feed_dict = {x_tensor: X_test, y_: y_train, keep_prob: 1.0})
    pdb.set_trace()

    #print('Accuracy given NN approach %0.2f \n' %(100*test_acc))

    death_test_pids = [102.0, 1050.0, 1107.0, 1115.0, 1171.0]
    for pid in test_death_pids: 
      model1pred(pid, x, y_, y_train, h_fc1, keep_prob, train_embedding)

    return None


def return_patient_centers(pids, x, y_, y_train, h_fc1, keep_prob, train_embedding):
  center_list = []
  for pid in pids:
    aa = get_all_adjacent_beats(pid)
    embedded_signal = h_fc1.eval(feed_dict = {x_tensor: aa, y_: y_train, keep_prob: 1.0})
    center = np.mean(embedded_signal, axis=1)
    center_list.append(center)
  return np.array(center_list)

def model1pred(pid, threshold, x, y_, y_train, h_fc1, keep_prob, train_embedding, plot=False):
    aa = get_all_adjacent_beats(pid)
    aa = standardize_ts_lengths(aa, PADDED_LENGTH)
    aa = np.swapaxes(aa, 1, 2)
    embedded_signal = h_fc1.eval(feed_dict = {x_tensor: aa, y_: y_train, keep_prob: 1.0})
    if plot:
      tsne.debug_plot(embedded_signal, np.ones((embedded_signal.shape[0])))
    return model_1_prediction(train_embedding, y_train, embedded_signal, threshold)

def get_upper_quartile_odds_ratio(test_death_ids, test_normal_ids, x, y_, y_train, h_fc1, keep_prob, train_embedding):

  def model1deathpred(pid):
      aa = get_all_adjacent_beats(pid)
      aa = standardize_ts_lengths(aa, PADDED_LENGTH)
      aa = np.swapaxes(aa, 1, 2)

      embedded_signal = h_fc1.eval(feed_dict = {x_tensor: aa, y_: y_train, keep_prob: 1.0})
      return percentage_death(train_embedding, y_train, embedded_signal)

  labels = [1 for pid in test_death_ids]
  labels.extend([0 for pid in test_normal_ids])

  all_ids = np.concatenate([test_death_ids, test_normal_ids])
  death_percentages = [xx for xx in [model1deathpred(pid) for pid in test_death_ids] if x is not None]
  normal_percentages = [xx for xx in [model1deathpred(pid) for pid in test_normal_ids] if x is not None]
  cutoff = np.percentile(death_percentages+normal_percentages, 75)

  n_death_correct = float(len([p for p in death_percentages if p > cutoff]))
  n_death_incorrect = float(len(death_percentages)-n_death_correct)
  n_normal_incorrect = float(len([p for p in normal_percentages if p > cutoff]))
  n_normal_correct = float(len(normal_percentages)-n_normal_incorrect)

  high_risk_odds = n_death_correct/(n_death_correct+n_normal_incorrect)
  low_risk_odds = n_death_incorrect/(n_death_incorrect+n_normal_correct)

  high_risk_patients = np.concatenate([test_death_ids[np.where(death_percentages > cutoff)[0]],test_normal_ids[np.where(normal_percentages > cutoff)[0]]])
  low_risk_patients = np.concatenate([test_death_ids[np.where(death_percentages <= cutoff)[0]],test_normal_ids[np.where(normal_percentages <= cutoff)[0]]])


  print "High risk: ", high_risk_odds
  print "Low risk: ", low_risk_odds
  print "Ratio: ", high_risk_odds/float(low_risk_odds)
  pdb.set_trace()

  return high_risk_patients, low_risk_patients


def run_cox_proportions(high_risk_patients, low_risk_patients):
  cph = CoxPHFitter()

  observations = generate_cox_dataframe(high_risk_patients, low_risk_patients)
  cox_model = cph.fit(observations, duration_col='CV_death_days', event_col='CV_death')
  cox_model.print_summary()

def generate_cox_dataframe(high_risk_patients, low_risk_patients):
  outcomes = pd.DataFrame.from_csv("./models/supervised/patient_outcomes.csv")
  outcomes['jiffy_risk'] = 0
  outcomes.loc[outcomes['id'].isin(high_risk_patients),'jiffy_risk'] = 1.0

  all_patients = np.concatenate([high_risk_patients, low_risk_patients])
  relevant_patients = outcomes[outcomes['id'].isin(all_patients)]
  return relevant_patients

def calculate_hazard_ratio(threshold, test_death_ids, test_normal_ids, x, y_,  y_train, h_fc1, keep_prob, train_embedding):
  n_death_1 = 0.0
  n_1 = 0.0
  n_death_0 = 0.0
  n_0 = 0.0

  for pid in test_death_ids:
    result = model1pred(pid, threshold, x, y_, y_train, h_fc1, keep_prob, train_embedding)
    if result == 0:
      n_death_0 += 1
      n_0 += 1
    else:
      n_death_1 += 1
      n_1 += 1

  for pid in test_normal_ids:
    result = model1pred(pid, threshold, x, y_, y_train, h_fc1, keep_prob, train_embedding)
    if result == 0:
      n_0 += 1
    else:
      n_1 += 1

  print "Hazard of high risk: ", n_death_1/n_1 
  print "Hazard of low risk: ", n_death_0/n_0
  print "Percentage of correctly recognized within death: ", (n_death_1)/(n_death_1+n_death_0)
  print "Hazard ratio", (n_death_1/n_1)/(n_death_0/n_0)
  pdb.set_trace()

if __name__ == "__main__":
  test_model(sys.argv[1], .2, num_fc_1)
