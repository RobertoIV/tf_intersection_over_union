import tensorflow as tf

def pred2coord(x,y,w,h):
	"""
	Converts from centroid/shape to 2-points coordinates.
	Inputs:
		x,y,w,h (tf.Tensor): Contains the location of the bounding box
			as in form of the centroid (xy) and its shape (w,h). Shape
			of each one of them: [?,13,13,5]
	Output:
		coord (tf.Tensor): Returns the 2-points coordinates of the bounding
			box containing the object. Shape [?,13,13,5,4] where '4' is 
			[y_top,x_left,y_bottom,x_right]
	"""
	centroid = tf.concat([tf.expand_dims(y,axis=-1),tf.expand_dims(x,axis=-1)],axis=-1)
	shape = tf.concat([tf.expand_dims(h,axis=-1),tf.expand_dims(w,axis=-1)],axis=-1)

	bbox_min = centroid - (shape/2)
	bbox_max = centroid + (shape/2)

	coord = tf.concat([bbox_min[...,0:1], bbox_min[...,1:2], bbox_max[...,0:1], bbox_max[...,1:2]], axis=-1)

	return(coord)
  
  def iou(bbox1,bbox2):
	"""
	Args:
        bbox (tf.Tensor): Contains the location of the bounding boxes 
                as centroid and shape -> [x,y,w,h]. Shape of tensor: for 
                YOLO it should be [?,13,13,5,4], but any other shape 
                should work. I.E. [?,4]
	"""
	A_area = tf.multiply(bbox1[...,2],bbox1[...,3])
	B_area = tf.multiply(bbox2[...,2],bbox2[...,3])

	bbox1 = pred2coord(bbox1[...,0],bbox1[...,1],bbox1[...,2],bbox1[...,3])
	bbox2 = pred2coord(bbox2[...,0],bbox2[...,1],bbox2[...,2],bbox2[...,3])
	"""
	bbox (tf.Tensor): Shape [?,13,13,5,4]
					[y_top,x_left,y_botom,x_right]
	"""
	# Here the reference frame changes. Y axis becomes positive for the upper
	# part. Normally with images the upper part "is" negative
	A_max = tf.stack([bbox1[...,2],bbox1[...,3]],axis=-1) # Upper Right
	A_min = tf.stack([bbox1[...,0],bbox1[...,1]],axis=-1) # Lower Left

	B_max = tf.stack([bbox2[...,2],bbox2[...,3]],axis=-1) # Upper Right
	B_min = tf.stack([bbox2[...,0],bbox2[...,1]],axis=-1) # Lower Left

	inters_max = tf.minimum(A_max,B_max) # Upper Right
	inters_min = tf.maximum(A_min,B_min) # Lower Left

	# Check if they indeed have an union
	hw_inters = tf.maximum(inters_max-inters_min,tf.constant(0.))
	
	inters_area = tf.multiply(hw_inters[...,0],hw_inters[...,1])
	area_union = tf.subtract(tf.add(A_area,B_area),inters_area)
	epsilon = 1e-30
	iou = tf.divide(inters_area,tf.add(area_union,tf.constant(epsilon)))

	return(iou)
