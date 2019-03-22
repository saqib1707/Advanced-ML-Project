import tensorflow as tf
import numpy as np
import datahelpers
import random

# hyperparameters
alpha = 1.0
margin = 5.0
batch_size = 100
learning_rate = 1e-4
num_iteration = 350

def get_next_batch(step,num_fullbatch,num_training_ex):
	step = step%(num_fullbatch+1)
	if step%num_fullbatch == 0 and step!=0:
		lst = range(step*batch_size,num_training_ex)
		return lst+range(0,batch_size-len(lst))
	return range(step*batch_size,(step+1)*batch_size)

def datapreprocessing(train_x,test_x):
	mean_x = np.mean(train_x,axis=0)
	std_x = np.std(train_x,axis=0)
	train_x = (train_x - mean_x)/std_x
	test_x = (test_x - mean_x)/std_x
	return train_x,test_x

def weight_variable(shape):
	initial = tf.truncated_normal(shape=shape, mean=0.0, stddev=1.0, dtype=tf.float32)
	return tf.Variable(initial)

def bias_variable(shape, value=0.5):
	initial = tf.constant(value, shape=shape)
	return tf.Variable(initial)

def training(train_x, train_y, test_x, test_y):
	# start_time = time.time()
	[num_training_ex,num_features] = train_x.shape
	classes = np.unique(train_y)
	num_classes = classes.shape[0]
	num_test_ex = test_x.shape[0]
	num_hidden1_units = num_features/4
	num_fullbatch = int(num_training_ex/batch_size)

	input_placeholder = tf.placeholder(tf.float32, shape=[None,num_features], name='image_features') #[batch_size,3072]
	label_placeholder = tf.placeholder(tf.int32, shape=[None], name='ground_truth_labels')   # [batch_size]
	predicted_labels = tf.placeholder(tf.int32, shape=[None], name='predicted_labels')
	print('Placeholders added')

	# Fully connected layer
	w1 = weight_variable([num_features, num_hidden1_units])
	b1 = bias_variable([num_hidden1_units])
	out_hidden1 = tf.nn.relu(tf.add(tf.matmul(input_placeholder, w1), b1))
	w2 = weight_variable([num_hidden1_units, num_classes])
	b2 = bias_variable([num_classes])
	output = tf.add(tf.matmul(out_hidden1, w2),b2, name='output')   # [batch_size, num_classes]

	# Hinge loss implementation
	max_margin_loss = tf.maximum(0.0,tf.add(margin,[[(-output[i,label_placeholder[i]] + output[i,j])
						for j in range(num_classes)] for i in range(batch_size)]))
	data_loss = tf.reduce_mean(tf.reduce_sum(max_margin_loss,reduction_indices=1)-margin)
	weight_loss = alpha*(tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
	loss = data_loss + weight_loss
	print('Loss function added to the tensorflow graph')

	train = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels,label_placeholder), tf.float32),name='accuracy')

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		lst = range(num_training_ex)
		random.shuffle(lst)
		train_x = train_x[lst]
		train_y = train_y[lst]

		for step in range(num_iteration):
			indices = get_next_batch(step,num_fullbatch,num_training_ex)

			train_batch_x = train_x[indices]             # [batch_size,3072]
			train_batch_y = train_y[indices]              # [batch_size]

			_, batch_loss = sess.run([train,loss]																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																									,feed_dict={input_placeholder:train_batch_x,label_placeholder:train_batch_y})

			print('Step:{}, Loss:{}'.format(step, batch_loss))

		# print('Calculating the training accuracy')
		# training_output = sess.run(output,feed_dict={input_placeholder:train_x})
		# prediction = np.argmax(training_output,axis=1)
		# training_accuracy = sess.run(accuracy,feed_dict={label_placeholder:train_y,predicted_labels:prediction})
		# print("Training Accuracy :",training_accuracy)

if __name__ == '__main__':
	dataset = datahelpers.data_cifar10()
	dataset['training_features'],dataset['test_features'] = datapreprocessing(dataset['training_features'],dataset['test_features'])
	training(dataset['training_features'],dataset['training_labels'],dataset['test_features'],dataset['test_labels'])