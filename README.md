# Tiny_Torch

This project contains three dictionaries including 
1. 'node_op' includes files about how to construct a computational graph that can do auto differentiation efficiently using TVM, unit in node
2. 'deep_op' includes files about how to construct deep learning layers that can do auto differentiation, unit in layer
3. 'basic_model' includes files about how to use files in 'deep_op' to construct deep learning model from scrach
4. 'other' include files about other types of deep network from scrach

Note that these files are not entirely correlated with each other. They come from different projects. 'node_op' is an independent project which means it is not used by other two projects. 'deep_op' and 'basic_model' are in a same project which means 'basic_model' used the layers defined in 'deep_op'. 'other' contains more advanced models that written from scrach and didn't depand on other two projects.
