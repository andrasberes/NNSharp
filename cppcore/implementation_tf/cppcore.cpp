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
	
	std::cout << "Hi!" << std::endl;
	
	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;

	//init
	const std::vector<int64_t> dimensions = { 2, 2 };	//{depth, row, column}
	std::size_t data_size = sizeof(int64_t);
	for (auto i : dimensions) {
		data_size *= i;
	}
	const std::vector<int64_t> data = {
		1, 2,
		3, 4};

	TF_Tensor* myTensor = TF_AllocateTensor(TF_INT64, dimensions.data(), 
		static_cast<int64_t>(dimensions.size()), data_size);
	
	if (myTensor != nullptr && TF_TensorData(myTensor) != nullptr) {
		std::memcpy(TF_TensorData(myTensor), data.data(), std::min(data_size, TF_TensorByteSize(myTensor)));
	}

	//return values 
	TF_DataType retDataType = TF_TensorType(myTensor);
	int retDimension = TF_NumDims(myTensor);
	const auto retDataBuffer = static_cast<int64_t*>(TF_TensorData(myTensor));


	//check databuffer
	for (std::size_t i = 0; i < data.size(); ++i) {
		if (retDataBuffer[i] != data[i]) {
			std::cout << "Element: " << retDataBuffer[i] << " does not match" << std::endl;
		} else {
			std::cout << "Element: " << retDataBuffer[i] << " matches. VERY GOOD!" << std::endl;
		}
	}
	std::cout << "Dimension: " << retDimension << std::endl;
	



	//Load graph from file
	TF_Graph* graph = LoadGraphDef("tensorflow_inception_graph.pb");
	if (graph == nullptr) {
		std::cout << "Can't load graph" << std::endl;
	}

	//Load graph's first operation
	size_t pos = 0;
	TF_Operation* operation = TF_GraphNextOperation(graph, &pos);
	if (operation == nullptr) {
		std::cout << "Can't init first operation" << std::endl;
	} else {
		std::cout << "Success: got first operation" << std::endl;
	}
	
	//Load operation by name
	TF_Input oper2 = { TF_GraphOperationByName(graph, "matmul"), 0 };
	if (oper2.oper == nullptr) {
		std::cout << "Can't load operation by name" << std::endl;
	} else {
		std::cout << "Success: loaded operation by name" << std::endl;
	}


	//deinit
	TF_DeleteTensor(myTensor);

	std::getchar();
	return 0;
}
