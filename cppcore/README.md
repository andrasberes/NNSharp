# Highlevel C++ API for deep learning

## Vision

The goal of this subproject is to create a high level C++ API which wraps the Tensorflow C API in a Keras-like manner with 
some extensions. This can have a couple of major advantages:

* It is easier to bind it to other languages and get a high level API in them without further implementations.
* Performance loss can be avoided due to the C++ implementation behind the API.
* Tensorflow does not need to be compiled alltogether, because the API uses only the Tensorflow DLL.

Note, that this subproject can be treated independently from NNSharp.

## Third party components

### Tensorflow C API
 TensorFlow provides a C API that can be used to build bindings for other languages. The API is defined in c_api.h and designed for simplicity 
and uniformity rather than convenience.
Tensorflow's main objects are Tensors, Graphs, and Sessions. The C API offers operations and methods which can be performed on these objects. 
Current version the project supports is 1.9.0, but it can be downloaded from the following link: 
https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.9.0.zip

### Protobuf
Protocol buffers are a flexible, efficient, automated mechanism for serializing structured data – think XML, but smaller, faster, and simpler. 
You define how you want your data to be structured once, then you can use special generated source code to easily write and read your structured 
data to and from a variety of data streams and using a variety of languages.

How do they work?
You specify how you want the information you're serializing to be structured by defining protocol buffer message types in .proto files. Each protocol buffer 
message is a small logical record of information, containing a series of name-value pairs.


## Principle

The Highlevel C++ API can help you create any neural network graphs in C++. After the developer designs a graph, configures its nodes, its inputs and operations,
 the constructed entity can be saved as a so called protobuf files, which can serve as an input for the Tensorflow library. After loading the graph, the developer needs 
 to define input data, and can carry out the usual neural network operations on the created object (from basic algorithms to classification and regression). 


## Implementation process 

First I created a Visual Studio Solution, where I added the corresponding Tensorflow C API header file. It contains all the structure, enum and function
definitions which are exported from an additional .lib file. This .lib file was created from the relevant version of Tensorflow DLL. The details of how one can extract lib from 
a DLL file can be found on the following link: 
* https://adrianhenke.wordpress.com/2008/12/05/create-lib-file-from-dll/

After adding the .lib file to the linker's additional dependencies, the provided example code in cppcore.cpp file can be compiled. The corresponding Tensorflow 
DLL needs to be placed next to the compiled binary file in order to run or debug the application. While I was trying out the library I realized the importance of protobuf files. 
Graphs can be saved as text and binary protobuf formats. Google Protobuf proposes a possibility to build your own message structure, and generate source code of any 
language. So I created the adequate message structures, and generated C++ files. After adding the headers to the project, all the necessary objects and functions are ready to use. 

## Progress
This is an initial version of the C++ API for Tensorflow framework. Saving a graph as a protobuf file is currently under construction, and not safe for work. 