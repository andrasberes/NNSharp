#include <vector>
#include <iostream>
#include <stdio.h>
#include <algorithm>

#define COMPILER_MSVC
extern "C" {
	#include "tensorflow.h"		//Tensorflow C API
}

void tf_test() {
	
	std::cout << "Hi!" << std::endl;
	
	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;
	//TF_Buffer* ops = TF_GetAllOpList();

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
	
	//deinit
	TF_DeleteTensor(myTensor);
}


int main(int argc, char *argv[]) {
	tf_test();
	std::getchar();
	return 0;
}

// DLL test
/*
core::tensor::Tensor::Tensor(DataType data_type, int dim, int* shape) {

	if (data_type == DataType::Integer) {

		// Calculate the size.
		int length = 1;
		for (int i = 0; i < dim; ++i) {
			length *= shape[i];
		}

		values = new int[length];
		this->dim = dim;
		this->shape = shape;
	}
	else
		values = nullptr;
}

core::tensor::Tensor::~Tensor() {
	if (values != nullptr)
		delete[] values;
}

core::tensor::TensorInteger::TensorInteger(int size):
	Tensor(DataType::Integer, 1, &size){
}

void core::tensor::TensorInteger::Get(int indices, int* value_out) const {
	const char* version = TF_Version();
	std::cout << version[0] << version[1] << version[2] << version[3] << version[4] << std::endl;
	*value_out = version[1];
}

void core::tensor::TensorInteger::Set(int idx, int value_in){

}

core::tensor::TensorInteger* core::tensor::create_tensor_integer(int size)
{
	return new TensorInteger(size);
}

int core::tensor::tensor_integer_get(TensorInteger * tensor_in, int idx)
{
	int retVal = 0;
	tensor_in->Get(idx, &retVal);

	return retVal;
}


*/
