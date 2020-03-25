#!/usr/bin/env python
# coding: utf-8

# In[16]:


import tensorflow as tf


# In[17]:


x_train = [1,2,3]
y_train = [1,2,3]


# In[18]:


# Try to find values for W and b that compute y_data = W * x_data + b
W = tf.Variable(tf.random_uniform([1],-1.0,1.0))
b = tf.Variable(tf.random_uniform([1],-1.0,1.0))


# In[20]:


X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)


# In[21]:


hypothesis = W * X + b


# In[22]:


cost = tf.reduce_mean(tf.square(hypothesis -Y))


# In[23]:


a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)


# In[24]:


init = tf.global_variables_initializer()


# In[25]:


sess = tf.Session()
sess.run(init)


# In[29]:


for step in range(2001):
    sess.run(train, feed_dict={X:x_train, Y:y_train})
    if step % 20 == 0:
        print(step, sess.run(cost,feed_dict={X:x_train, Y:y_train}), sess.run(W), sess.run(b))


# In[30]:


print(sess.run(hypothesis, feed_dict={X:5}))
print(sess.run(hypothesis, feed_dict={X:0.5}))


# In[ ]:




