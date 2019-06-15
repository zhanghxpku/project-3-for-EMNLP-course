# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 08:42:35 2019

@author: 38192
"""

import tensorflow as tf

checkpoint_file = tf.train.latest_checkpoint('./results/semantic_parsing/checkpoint/')
print(checkpoint_file)

reader = tf.train.NewCheckpointReader(checkpoint_file)
#print(reader.debug_string().decode("utf-8"))
var_to_shape_map = reader.get_variable_to_shape_map() 
for key in var_to_shape_map: 
    print("tensor_name: ", key, 'tensor_shape: ',var_to_shape_map[key])   # 打印变量名
#    if key == 'w':
#        print(reader.get_tensor(key)) # 打印变量值 