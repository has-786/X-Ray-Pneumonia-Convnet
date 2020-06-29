import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow.compat.v1 as tf
import cv2
import glob
import matplotlib.pyplot as plt


tf.disable_v2_behavior()
x = tf.placeholder(tf.float32, [None, 200,200,1])
y = tf.placeholder(tf.float32, [None, 3])

filenames = glob.glob("../input/chest-xray-pneumonia/chest_xray/train/NORMAL/*.jpeg")
images = np.array([ cv2.resize( cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY),(200,200) ) for img in filenames])
#cv2.imshow("image",images[0])


x_train=np.array([images[0]])
y_train=np.array([[1,0,0]])

print(images[0].shape)
for i in range(1,500):
    x_train=np.append(x_train,[images[i]],axis=0)
    y_train=np.append(y_train,[[1,0,0]],axis=0)

filenames = glob.glob("../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/*.jpeg")
images = np.array([ cv2.resize( cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY),(200,200) ) for img in filenames])

for i in range(0,700):
    x_train=np.append(x_train,[images[i]],axis=0)
    if "virus" in filenames[0]:
        y_train=np.append(y_train,[[0,0,1]],axis=0)
    else:
        y_train=np.append(y_train,[[0,1,0]],axis=0)       
        
filenames = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/NORMAL/*.jpeg")
#print(filenames)
images = np.array([ cv2.resize( cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY),(200,200) ) for img in filenames])

#print(images[0].shape)

x_test=np.array([images[0]])
y_test=np.array([[1,0,0]])

for i in range(1,100):
    x_test=np.append(x_test,[images[i]],axis=0)
    y_test=np.append(y_test,[[1,0,0]],axis=0)

filenames = glob.glob("../input/chest-xray-pneumonia/chest_xray/test/PNEUMONIA/*.jpeg")
images = np.array([ cv2.resize( cv2.cvtColor(cv2.imread(img),cv2.COLOR_BGR2GRAY),(200,200) ) for img in filenames])

for i in range(0,200):
    x_test=np.append(x_test,[images[i]],axis=0)
    if "virus" in filenames[0]:
        y_test=np.append(y_test,[[0,0,1]],axis=0)
    else:
        y_test=np.append(y_test,[[0,1,0]],axis=0)



# Python optimisation variables
learning_rate = 0.0001
epochs = 5
batch_size = 50



x_train = np.reshape(x_train, (-1,200,200, 1))
x_test = np.reshape(x_test, (-1, 200, 200, 1))


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)




def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
    # setup the filter input shape for tf.nn.conv_2d
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,num_filters]

    # initialise weights and bias for the filter
    weights = tf.Variable(tf.truncated_normal(conv_filt_shape, stddev=0.03),
                                      name=name+'_W')
    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name+'_b')
    input_data = tf.cast(input_data, tf.float32)

    # setup the convolutional layer operation
    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    # add the bias
    out_layer += bias
    
    # apply a ReLU non-linear activation
    out_layer = tf.nn.relu(out_layer)
    print("2nd Conv Layer: ",out_layer.shape)
    # now perform max pooling
    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides,padding='SAME')
    print("2nd Pooling Layer: ",out_layer.shape)

    return out_layer


# Python optimisation variables
learning_rate = 0.0001
epochs = 5
batch_size = 50


# create some convolutional layers
layer1 = create_new_conv_layer(x, 1, 32, [5, 5], [2, 2], name='layer1')
layer2 = create_new_conv_layer(layer1, 32, 64, [5, 5], [2, 2], name='layer2')

flattened = tf.reshape(layer2, [-1, 50 * 50 * 64])
wd1 = tf.Variable(tf.truncated_normal([50 * 50 * 64, 1000], stddev=0.03), name='wd1')
bd1 = tf.Variable(tf.truncated_normal([1000], stddev=0.01), name='bd1')
dense_layer1 = tf.matmul(flattened, wd1) + bd1
dense_layer1 = tf.nn.relu(dense_layer1)

wd2 = tf.Variable(tf.truncated_normal([1000, 3], stddev=0.03), name='wd2')
bd2 = tf.Variable(tf.truncated_normal([3], stddev=0.01), name='bd2')
dense_layer2 = tf.matmul(dense_layer1, wd2) + bd2
y_ = tf.nn.softmax(dense_layer2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_layer2, labels=y))
# add an optimiser
optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# setup the initialisation operator
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    # initialise the variables
    sess.run(init_op)
    total_batch = int(len(y_train) / batch_size)
    print("Total Batch Per Epoch: ",total_batch)
    for epoch in range(epochs):
        print(epoch)
        avg_cost = 0
        t=0
        for i in range(total_batch):
            print("Batch ",i,"of Epoch ",epoch)
            batch_x, batch_y =x_train[t:t+batch_size],y_train[t:t+batch_size]
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
            t+=batch_size
        test_acc = sess.run(accuracy,feed_dict={x:x_test, y: y_test})
        print(test_acc)
       # print("Epoch:", (epoch + 1), "cost =", "{}".format(avg_cost), "test accuracy: {}".format(test_acc))

    print("\nTraining complete!")
    print(sess.run(accuracy, feed_dict={x:x_test, y:y_test}))
