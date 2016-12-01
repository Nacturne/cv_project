import tensorflow as tf

x = tf.constant([[1,2.2,3],[4.000,5,6]])
y = tf.constant([[1,1,0.834534],[0.242423434,0,1]])
out = tf.add(x,y)
out2 = x+y
sess = tf.Session()

print sess.run(out)
print sess.run(out2)