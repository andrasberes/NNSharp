import tensorflow as tf
from google.protobuf import text_format

with open('myfile.pbtxt') as f:
  txt = f.read()
gdef = text_format.Parse(txt, tf.GraphDef())

tf.train.write_graph(gdef, '.', 'myfile.pb', as_text=False)