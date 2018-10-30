#include <vector>
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <string>

#define COMPILER_MSVC
extern "C" {
#include "tensorflow.h"		//Tensorflow C API
}
#include "../utils.hpp"

using namespace utils;

int main(int argc, char *argv[]) {
	
	/*--------------------VERSION---------------------*/
	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;

	/*--------------------TENSOR INIT-----------------*/
	//Tensor
	const std::vector<std::int64_t> input_dims = { 1 };
	const std::vector<int32_t> input_data = { 2 };
	TF_Tensor* myTensor = CreateTensor(TF_INT32, input_dims, input_data);
	const std::vector<TF_Tensor*> input_tensors = { myTensor, myTensor };

	//validate tensor data
	TF_DataType retDataType = TF_TensorType(myTensor);
	int retDimension = TF_NumDims(myTensor);
	const auto retDataBuffer = static_cast<int32_t*>(TF_TensorData(myTensor));
	//check databuffer
	for (std::size_t i = 0; i < input_data.size(); ++i) {
		if (retDataBuffer[i] != input_data[i]) {
			std::cout << "Element: " << retDataBuffer[i] << " does not match" << std::endl;
		} else {
			std::cout << "Element: " << retDataBuffer[i] << " matches. VERY GOOD!" << std::endl;
		}
	}
	std::cout << "Dimension: " << retDimension << std::endl;
	/*-------------------------------------------------*/


	//Load graph from file
	TF_Graph* graph = LoadGraphDef("example_graph1.pb");
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
		std::getchar();
		return -1;
	}

	//Load operation by name
	TF_Output out_op = { TF_GraphOperationByName(graph, "op_add"), 0 };
	if (out_op.oper == nullptr) {
		std::cout << "Can't load operation by name" << std::endl;
		return -3;
	} else {
		std::cout << "Success: loaded operation by name" << std::endl;
	}

	//Input and Output tensor
	TF_Tensor* output_tensor = nullptr;
	TF_Output input_ops[2];
	input_ops[0] = { TF_GraphOperationByName(graph, "a"), 0 };
	if (input_ops[0].oper == nullptr) {
		std::cout << "Can't init input a" << std::endl;
		return 2;
	} else {
		std::cout << "Success: loaded input a" << std::endl;
	}
	input_ops[1] = { TF_GraphOperationByName(graph, "b"), 0 };
	if (input_ops[1].oper == nullptr) {
		std::cout << "Can't init input b" << std::endl;
		return 3;
	} else {
		std::cout << "Success: loaded input b" << std::endl;
	}


	//Session run
	bool success = RunSession(graph, input_ops, input_tensors.data(), 2, &out_op, &output_tensor, 1);
	if (success) {
		const auto out_data = static_cast<int32_t*>(TF_TensorData(output_tensor));
		std::cout << "Output vals: " << out_data[0] << ", " << out_data[1] << ", " << out_data[2] << ", " << out_data[3] << std::endl;
	} else {
		std::cout << "Error running session";
		return 2;
	}

	//Creating binary file from txt
	utils::converter::txt2binary("example_graph1.txt", "myExample_graph_1.pb");

	//deinit
	DeleteTensors(input_tensors);
	TF_DeleteGraph(graph);

	
	std::cout << "End of running" << std::endl;
	std::getchar();
	return 0;
}
