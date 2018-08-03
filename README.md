# Intersection Over Union
Method to do intersection over union with N number of inputs using tensorflow.

The function requires two tensors, each one containing the bounding boxes where the last dimension has 4 elements -> [Xc,Yc,W,H].
Xc: X position of the centroid.
Yc: Y position of the centroid
W: Width of its shape.
H: Height of its shape.

# How to use it
The example code is ```runme.py```.

First import the function that do all the work.
```
import utils as ut
```

Next, indicate which are the bounding boxes to be computed.
Here I am going to use the shape for YOLO Object Detection.
```
# Here we are going to introduce the bounding box 1
bb1 = tf.placeholder(tf.float32,shape=[None,13,13,5,4])
# Here we are going to introduce the bounding box 2
bb2 = tf.placeholder(tf.float32,shape=[None,13,13,5,4])
```

Then, create the graph that will do the IoU.
```
iou_res = ut.iou(bb1,bb2)
```

And the session to run all.
```
sess = tf.Session()
```

Lets createthe values to compute
```
a = np.zeros([1,13,13,5,4])
a[0,0,0,0] = [0.5,0.5,1,1] # Square with centroid at (0.5,0.5)
                           # and shape (1,1) -> (w,h)
b = np.zeros([1,13,13,5,4])
b[0,0,0,0] = [-0.5,-0.5,2,2] # Square with centroid at (-0.5,-0.5)
                             # and shape (2,2) -> (w,h)
```

And compute everything.
```
iou = sess.run(iou_res,feed_dict={bb1:a,bb2:b})
print(iou[0,0,0,0]) # Prints the IoU 
```
