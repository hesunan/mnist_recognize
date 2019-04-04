import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#MNIST数据集相关常数
INPUT_NODE = 784    #输入层节点数，即图片像素
OUTPUT_NODE = 10    #输出层节点数，即类别个数。MNIST数据集中需要区分0~9这十个数

#配置神经网络参数
LAYER1_NODE = 500   #隐藏层节点数。此处使用有500个节点的一个隐藏层的网络结构
BATCH_SIZE = 100    #一个训练batch中训练数据个数
                    #数字越小，训练过程越接近随机梯度下降；数字越大越接近梯度下降
LEARNING_RATE_BASE = 0.8        #基础学习率
LEARNING_RATE_DECAY = 0.99      #学习率的衰减率
REGULARIZATION_RATE = 0.0001    #描述模型复杂度的正则化项在损失函数中的系数
TRAINING_STEPS = 30000          #训练轮数
MOVING_AVERAGE_DECAY = 0.99     #滑动平均衰减率


def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    #当没有提供滑动平均类时，直接使用参数当前的取值
    if avg_class == None:
        #计算隐藏层前向传播结果，使用ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        return tf.matmul(layer1, weights2) +biases2

    else:
        #首先使用avg_class.average函数来计算得出变量的滑动平均值
        #然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) +avg_class.average(biases2)

#训练模型
def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    #生成隐藏层参数
    weights1 = tf.Variable(
        #正态分布，标准差为0.1
        tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
    #生成输出层参数
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    
    y = inference(x, None, weights1, biases1, weights2, biases2)   
    global_step = tf.Variable(0, trainable = False)   
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)   
    variables_averages_op = variable_averages.apply(
        tf.trainable_variables())    
    average_y = inference(
        x, variable_averages, weights1, biases1, weights2, biases2)
    #计算交叉熵作为刻画预测值和真实值之间差距的损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))
    #计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    #计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    #计算模型的正则化损失。一般只计算NN边上权重的正则化损失，而不使用偏置项
    regularization = regularizer(weights1) + regularizer(weights2)
    #总损失等于交叉熵损失和正则化损失的和
    loss = cross_entropy_mean + regularization
  
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples/BATCH_SIZE, LEARNING_RATE_DECAY)

    #使用tf.train.GradientDescentOptimizer优化算法来优化损失函数
    #此处损失函数包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate)\
                .minimize(loss, global_step=global_step)

    
    #为了一次完成多个操作，TensorFlow提供了tf.control_dependencies和tf.group两种机制
    #下面两行程序和train_op = tf.group(train_step,variables_averages_op)是等价的
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))    
    #此运算首先将一个布尔型的数值转换为实数型，然后计算平均值
    #此平均值即为模型在这一组数据上的正确率
    accuracy =  tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #初始化会话并开始训练
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        #准备验证数据
        #在NN的训练过程中通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {x: mnist.validation.images,
                        y_: mnist.validation.labels}

        #准备测试数据
        #在实际应用中，此部分数据训练时不可见，只是作为模型优劣的最后评价标准
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        #迭代训练神经网络
        for i in range(TRAINING_STEPS):
            #每1000轮输出一次在验证数据集上的测试结果
            if i % 1000 == 0:
                #计算滑动平均模型在验证数据上的结果
                #因为MNIST数据集比较小，故一次可以处理所有的验证数据
                #当NN模型比较复杂或者验证数据比较大时，太大的batch会导致计算时间过长甚至发生内存溢出的错误
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training steps, validation accuracy "
                      "using average model is %g" % (i, validate_acc))

                #产生此轮使用的一个batch训练数据，并运行训练过程
                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

        #训练结束后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training steps, test accuracy "
              "using average model is %g" % (TRAINING_STEPS, test_acc))

#主程序入口
def main(argv=None):
    #声明处理MNIST数据集的类
    mnist = input_data.read_data_sets("MNIST_data", one_hot = True)
    train(mnist)

#TensorFlow提供的主程序入口，tf.app.run会调用上述定义的main函数
if __name__ == '__main__':
    main()

    
