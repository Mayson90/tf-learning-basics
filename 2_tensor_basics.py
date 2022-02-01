import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

# GPU activate
# physical_devices = tf.config.list_physical_devices("GPU")
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Initialization of Tensors
x = tf.constant(4, shape=(1, 1), dtype=tf.float32)
x = tf.constant([[1, 2, 3], [4, 5, 6]])

x = tf.ones((3, 3))
x = tf.zeros((4, 6))
x = tf.eye(8)
x = tf.random.normal((3, 3), mean=0, stddev=3)
x = tf.random.uniform((1, 3), minval=-1, maxval=4)
x = tf.range(start=1, limit=9, delta=2)
x = tf.cast(x, dtype=tf.float64)

# Mathematical Operations
x = tf.constant([1, 2, 3])
y = tf.constant([9, 9, 7])

z = tf.add(x, y)
z = x + y

z = tf.subtract(x, y)
z = x - y

z = tf.divide(x, y)
z = x / y

z = tf.multiply(x, y)
z = x * y

z = tf.tensordot(x, y, axes=1)
z = tf.reduce_sum(x*y, axis=0)

z = x ** 5

x = tf.random.normal((2, 3))
y = tf.random.normal((3, 4))
z = tf.matmul(x, y)

z = x @ y

# Indexing
x = tf.constant([0, 1, 1, 2, 3, 1, 2, 3])
# print(x[:])
# print(x[1:])
# print(x[1:3])
# print(x[::2])
# print(x[::-1])

indices = tf.constant([0, 6])
x_ind = tf.gather(x, indices)

x = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]])

# print(x[0, :])
# print(x[0:2, :])

# Reshaping
x = tf.range(9)
print(x)

x = tf.reshape(x, (3, 3))
print(x)

x = tf.transpose(x, perm=[1, 0])
print(x)
