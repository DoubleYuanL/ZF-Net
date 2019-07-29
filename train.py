import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import cnn_utils
import cv2
import pickle
#加载数据集
def load_dataset():
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig , classes = cnn_utils.load_dataset()
	return X_train_orig , Y_train_orig , X_test_orig , Y_test_orig ,classes
# def load(f_name):
# 	with open(f_name, "rb") as fo:
# 		data = pickle.load(fo, encoding = 'bytes')
# 		return data
# def load_data():
# 	train_data = load("cifar-10-batches-py/data_batch_1")
# 	data_x = np.array(train_data[b"data"]).reshape(-1, 3, 32, 32)
# 	data_y = np.array(train_data[b"labels"])

# 	x_train = data_x[0:8000,:]
# 	y_train = data_y[0:8000]
# 	print(x_train.shape)
# 	print(y_train.shape)

# 	x_test = data_x[8000:10000,:]
# 	y_test = data_y[8000:10000]
# 	print(x_test.shape)
# 	print(y_test.shape)

# 	y_train = y_train.reshape((1, y_train.shape[0]))
# 	y_test = y_test.reshape((1, y_test.shape[0]))

# 	name = load("cifar-10-batches-py/batches.meta")
# 	name = np.array(name[b"label_names"])
# 	print(name)
# 	return x_train,y_train,x_test,y_test,name
# load_data()
#初始化数据集
def init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig ):
	X_train = X_train_orig/255.
	X_test = X_test_orig/255.
	Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
	Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
	# print ("number of training examples = " + str(X_train.shape[0]))
	# print ("number of test examples = " + str(X_test.shape[0])) 
	# print ("X_train shape: " + str(X_train.shape)) 
	# print ("Y_train shape: " + str(Y_train.shape))
	# print ("X_test shape: " + str(X_test.shape))
	# print ("Y_test shape: " + str(Y_test.shape))
	return X_train, Y_train, X_test, Y_test

#创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
	Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
	keep_prob = tf.placeholder(tf.float32)
	return X,Y,keep_prob
#初始化参数
def init_parameters():
	tf.set_random_seed(1) #指定随机种子
	W1 = tf.get_variable("W1",[3,3,3,96], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b1 = tf.get_variable("b1",[96,], initializer=tf.zeros_initializer())
	W2 = tf.get_variable("W2",[3,3,96,256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b2 = tf.get_variable("b2",[256,], initializer=tf.zeros_initializer())
	W3 = tf.get_variable("W3",[3,3,256,384], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b3 = tf.get_variable("b3",[384], initializer=tf.zeros_initializer())
	W4 = tf.get_variable("W4",[3,3,384,512], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b4 = tf.get_variable("b4",[512], initializer=tf.zeros_initializer())
	W5 = tf.get_variable("W5",[3,3,512,256], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	b5 = tf.get_variable("b5",[256], initializer=tf.zeros_initializer())

	parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3, "W4": W4, "b4": b4, "W5": W5, "b5": b5} 

	return parameters
#前向传播
def forward_propagation(X, parameters,keep_prob):
	W1 = parameters['W1'] 
	b1 = parameters['b1'] 
	W2 = parameters['W2'] 
	b2 = parameters['b2'] 
	W3 = parameters['W3'] 
	b3 = parameters['b3']
	W4 = parameters['W4'] 
	b4 = parameters['b4']
	W5 = parameters['W5'] 
	b5 = parameters['b5']

	Z1 = tf.nn.bias_add(tf.nn.conv2d(X,W1,strides=[1,2,2,1],padding="VALID"),b1)
	A1 = tf.nn.relu(Z1)
	#lrn1 = tf.nn.lrn(A1, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn1")
	P1 = tf.nn.max_pool(A1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
	# print("p1"+str(P1.shape))

	Z2 = tf.nn.bias_add(tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME"),b2)
	A2 = tf.nn.relu(Z2)
	# lrn2 = tf.nn.lrn(A2, 4, bias=1.0, alpha=0.001/9, beta=0.75, name="lrn2")
	P2 = tf.nn.max_pool(A2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")

	Z3 = tf.nn.bias_add(tf.nn.conv2d(P2,W3,strides=[1,1,1,1],padding="SAME"),b3)
	A3 = tf.nn.relu(Z3)

	Z4 = tf.nn.bias_add(tf.nn.conv2d(A3,W4,strides=[1,1,1,1],padding="SAME"),b4)
	A4 = tf.nn.relu(Z4)

	Z5 = tf.nn.bias_add(tf.nn.conv2d(A4,W5,strides=[1,1,1,1],padding="SAME"),b5)
	A5 = tf.nn.relu(Z5)
	P5 = tf.nn.max_pool(A5,ksize=[1,3,3,1],strides=[1,2,2,1],padding="VALID")
	print(P5.shape)

	#f1
	Fa1 = tf.contrib.layers.flatten(P5)
	F1 = tf.contrib.layers.fully_connected(Fa1,128,activation_fn=tf.nn.relu)#None)#tf.nn.relu) tf.nn.sigmoid
	F1 = tf.nn.dropout(F1, keep_prob=keep_prob)

	F2 = tf.contrib.layers.fully_connected(F1,64,activation_fn=tf.nn.relu)
	F2 = tf.nn.dropout(F2, keep_prob=keep_prob)	

	Z6 = tf.contrib.layers.fully_connected(F2,6,activation_fn=None)

	return Z6
#计算loss
def compute_loss(Z6,Y):
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = Z6, labels = Y))
	return loss 
#定义模型
def model(X_train,Y_train,X_test,Y_test,learning_rate,num_epochs,minibatch_size,print_cost,isPlot):
	seed = 3
	# X_train = X_train.transpose(0,2,3,1)
	(m , n_H0, n_W0, n_C0) = X_train.shape
	n_y = Y_train.shape[1]
	costs = []

	X,Y,keep_prob = create_placeholder(n_H0, n_W0, n_C0, n_y)

	parameters = init_parameters()

	Z6 = forward_propagation(X, parameters,keep_prob)

	cost = compute_loss(Z6, Y)

	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

	init = tf.global_variables_initializer()

	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost = 0
			num_minibatches = int(m / minibatch_size) #获取数据块的数量
			#seed = seed + 1
			minibatches = cnn_utils.random_mini_batches(X_train,Y_train,minibatch_size,seed) 

			#对每个数据块进行处理
			for minibatch in minibatches:
				#选择一个数据块
				(minibatch_X,minibatch_Y) = minibatch
				#最小化这个数据块的成本
				_ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X, Y:minibatch_Y, keep_prob:0.5})

				#累加数据块的成本值
				minibatch_cost += temp_cost / num_minibatches

			#是否打印成本
			if print_cost:
				#每5代打印一次
				if epoch % 50 == 0:
					print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

			#记录成本
			if epoch % 1 == 0:
				costs.append(minibatch_cost)

		#数据处理完毕，绘制成本曲线
		if isPlot:
			plt.plot(np.squeeze(costs))
			plt.ylabel('cost')
			plt.xlabel('iterations (per tens)')
			plt.title("Learning rate =" + str(learning_rate))
			plt.show()
		#保存学习后的参数
		parameters = sess.run(parameters)
		print("参数已经保存到session。")
		saver.save(sess,"model/save_net.ckpt")


		# X_test = X_test.transpose(0,2,3,1)
		#开始预测数据
		## 计算当前的预测情况
		predict_op = tf.argmax(Z6,1)

		corrent_prediction = tf.equal(predict_op , tf.argmax(Y,1))

		##计算准确度
		accuracy = tf.reduce_mean(tf.cast(corrent_prediction,"float"))
		# print("corrent_prediction accuracy= " + str(accuracy))

		train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob:1.0})
		test_accuary = accuracy.eval({X: X_test, Y: Y_test, keep_prob:1.0})

		print("训练集准确度：" + str(train_accuracy))
		print("测试集准确度：" + str(test_accuary))

		return parameters

if __name__ == '__main__':
	X_train_orig , Y_train_orig , X_test_orig , Y_test_orig, name  = load_dataset()
	X_train,Y_train,X_test,Y_test = init_dataset(X_train_orig , Y_train_orig , X_test_orig , Y_test_orig)
	parameters = model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 200,minibatch_size=64,print_cost=True,isPlot=True)

