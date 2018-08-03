import numpy as np
import tensorflow as tf
import utils as ut

a = np.zeros([1,13,13,5,4])
a[0,0,0,0] = [0.5,0.5,1,1] # Square with centroid at (0.5,0.5)
                           # and shape (1,1) -> (w,h)
b = np.zeros([1,13,13,5,4])
b[0,0,0,0] = [-0.5,-0.5,2,2] # Square with centroid at (-0.5,-0.5)
                             # and shape (2,2) -> (w,h)

bb1 = tf.placeholder(tf.float32,shape=[None,13,13,5,4])
bb2 = tf.placeholder(tf.float32,shape=[None,13,13,5,4])

iou_res = ut.iou(bb1,bb2)

sess = tf.Session()

iou = sess.run(iou_res,feed_dict={bb1:a,bb2:b})

print(iou[0,0,0,0]) # IoU will be ~0.052631
