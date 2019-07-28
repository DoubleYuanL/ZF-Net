import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
import cnn_utils
import cv2

#创建占位符
def create_placeholder(n_H0, n_W0, n_C0, n_y):
	X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0], name = "X")
	Y = tf.placeholder(tf.float32, [None,n_y], name = "Y")
	return X,Y
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
def forward_propagation(X, parameters,is_train_or_prediction):
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
	D1 = tf.nn.dropout(F1, 0.8)
	F2 = tf.contrib.layers.fully_connected(D1,64,activation_fn=tf.nn.relu)

	D2 = tf.nn.dropout(F2, 0.5)	
	Z6 = tf.contrib.layers.fully_connected(D2,6,activation_fn=None)

	return Z6
def predict():
	X,_ = create_placeholder(64, 64, 3, 6)

	parameters = init_parameters()

	Z5 = forward_propagation(X, parameters,is_train_or_prediction=False)

	Z5 = tf.argmax(Z5,1)

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(init)
		saver.restore(sess,tf.train.latest_checkpoint("model/"))

		#use the sample picture to predict the unm
		sample = 1
		cam = 1
		if (sample):
			num = 0
			my_image = "sample/" + str(num) + ".jpg"	
			num_px = 64
			fname =  my_image 
			image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
			my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
			my_predicted_image = my_predicted_image.astype(np.float32)

			my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image})

			plt.imshow(image) 
			print("prediction num is : y = " + str(np.squeeze(my_predicted_image)))
			plt.show()
			num = num + 1
		elif(cam):# use the camera to predict the num
			cap = cv2.VideoCapture(0)
			while (1):
				num = 0
				ret, frame = cap.read()
				cv2.namedWindow("capture")
				cv2.imshow("capture", frame)
				k = cv2.waitKey(1) & 0xFF
				if  k == ord('s'):
					frame = cv2.resize(frame, (int(256), int(256)))
					cv2.imwrite("sample/cam/" + str(num)+".jpg", frame)

					my_image = "sample/cam/" + str(num) + ".jpg"	
					num_px = 64
					fname =  my_image 
					image = np.array(ndimage.imread(fname, flatten=False))#.astype(np.float32)
					my_predicted_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((1,64,64,3))/255
					my_predicted_image = my_predicted_image.astype(np.float32)

					my_predicted_image = sess.run(Z5, feed_dict={X:my_predicted_image})

					plt.imshow(image) 
					print("预测结果: y = " + str(np.squeeze(my_predicted_image)))
					plt.show()
					num = num + 1
				elif k == ord('q'):
					break
			cap.release()
			cv2.destroyAllWindows()

if __name__ == '__main__':
	predict()


