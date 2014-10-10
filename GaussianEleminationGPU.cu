#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "common.h"

const int TILE_SIZE = 256;          //Max no of threads in Block.
const int MAX_GRID_SIZE = 65535;    //Max no of Blocks  in a Grid.

// GPU Kernel-1 to perform row scaling.
__global__ void GaussianEliminationGPUKernelScaling(float* matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix, bool partialPivot, unsigned int row)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x;   //Thread id calculation

	if(threadId<numberOfColumns*numberOfRows){
		if ((threadId/numberOfColumns)==row)
			outputMatrix[threadId] = matrix[threadId] / matrix[numberOfColumns*row+row];
		else 
			outputMatrix[threadId] = matrix[threadId];
	}
}

// GPU Kernel-2 to perform the reduction in rows.
__global__ void GaussianEliminationGPUKernelReduction(float* Matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float* outputMatrix, bool partialPivot, unsigned int row)
{
	// Retrieve our coordinates in the block
	int blockId   = blockIdx.y * gridDim.x + blockIdx.x;			 	
	int threadId = blockId * blockDim.x + threadIdx.x;     //threadid calculation

	float pivot = Matrix[numberOfColumns*row+row];         //Calculates the pivot element for each row.
	if(threadId<numberOfColumns*numberOfRows){
		if ((threadId/numberOfColumns)!=row)
			outputMatrix[threadId] = Matrix[threadId]- Matrix[(threadId/numberOfColumns)*numberOfRows+row] * (Matrix[row*numberOfColumns + threadId%numberOfColumns]/pivot);
		else 
			outputMatrix[threadId] = Matrix[threadId];
		}
}

// GPU function for direct method Gross Jorden method.

bool GaussianEliminationGPU( float** matrix, unsigned int numberOfRows, unsigned int numberOfColumns, float** outputMatrix, bool partialPivot)
{
	// Error return value
	cudaError_t status;
	// Number of bytes in the matrix.
	int bytes = numberOfColumns * numberOfRows *sizeof(float);

	unsigned int numberOfRowsd, numberOfColumnsd;    //To be safe copy the elements too.
	numberOfColumnsd = numberOfColumns;
	numberOfRowsd = numberOfRows;

	// Pointers to the device arrays
	float *matrixd, *outputMatrixd;                    //input and output matrix
	// Allocate memory on the device to store each matrix
	cudaMalloc((void**) &matrixd, bytes);
	status = cudaGetLastError();              //To check the error
	if (status != cudaSuccess) {                     
		std::cout << "Kernel failed2: " << cudaGetErrorString(status) << 
		std::endl;
		cudaFree(matrixd);                     //Free call for memory
		cudaFree(outputMatrixd);               //Free call for memory
		return false;
	}

	cudaMalloc((void**) &outputMatrixd, bytes);
	status = cudaGetLastError();              //To check the error
	if (status != cudaSuccess) {                     
		std::cout << "Kernel failed2: " << cudaGetErrorString(status) << 
		std::endl;
		cudaFree(matrixd);                     //Free call for memory
		cudaFree(outputMatrixd);               //Free call for memory
		return false;
	}
	
	float *temp1 = matrixd;
	float *temp2 = outputMatrixd;

	// Copy the host input data to the device
	for (int i=0; i<numberOfRows; i++){
		cudaMemcpy((float *)temp1, matrix[i], numberOfColumns *sizeof(float), cudaMemcpyHostToDevice);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed3: " << cudaGetErrorString(status) << 
			std::endl;
			cudaFree(matrixd);                   //Free call for memory
			cudaFree(outputMatrixd);              //Free call for memory
			return false;
		}
		cudaMemcpy((float *)temp2, matrix[i], numberOfColumns *sizeof(float), cudaMemcpyHostToDevice);
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed4: " << cudaGetErrorString(status) << 
			std::endl;
			cudaFree(matrixd);                      //Free call for memory
			cudaFree(outputMatrixd);                //Free call for memory
			return false; 
		}
		temp1 += numberOfColumns;
		temp2 += numberOfColumns;
	}
	temp1= matrixd;
	temp2 = outputMatrixd;
	
	int size = numberOfColumns * numberOfRows;
	dim3 dimBlock(TILE_SIZE, 1);
	int gridx = 1;                      //Grid size calculation
	int gridy = 1;                      //Grid size calculation
	if(size/TILE_SIZE < MAX_GRID_SIZE)
		gridx = ceil((float)size/TILE_SIZE);            //Decide the grid size for input size.
	else{
		gridx = MAX_GRID_SIZE;
		gridy = ceil((float)size/(TILE_SIZE*MAX_GRID_SIZE));
	}

	dim3 dimGrid(gridx, gridy); // grid call.
	
	// Launch the kernel one-by-one
	int rowNo = 0; 
	for (rowNo=0; rowNo < numberOfColumns ;rowNo++){
		GaussianEliminationGPUKernelScaling<<<dimGrid, dimBlock>>>(matrixd, numberOfRowsd, numberOfColumnsd, outputMatrixd, partialPivot, rowNo);    //Calling kernel-1 for scaling
		cudaThreadSynchronize();                //Thread sync
		status = cudaGetLastError();
		if (status != cudaSuccess) {
			std::cout << "Kernel failed5: " << cudaGetErrorString(status) << 
			std::endl;
			cudaFree(matrixd);                   //Free call for memory
			cudaFree(outputMatrixd);             //Free call for memory
			return false;
		}
		
		GaussianEliminationGPUKernelReduction<<<dimGrid, dimBlock>>>(outputMatrixd, numberOfRowsd, numberOfColumnsd, matrixd, partialPivot, rowNo);       //Calling kernel-2 for reduction
		status = cudaGetLastError();     //Error check
		if (status != cudaSuccess) {
			std::cout << "Kernel failed6: " << cudaGetErrorString(status) << 
			std::endl;
			cudaFree(matrixd);                    //Free call for memory
			cudaFree(outputMatrixd);              //Free call for memory
			return false;
		}

		cudaThreadSynchronize();          //thread sync
	}

	// Check for errors
	status = cudaGetLastError();
	if (status != cudaSuccess) {
		std::cout << "Kernel failed7: " << cudaGetErrorString(status) << 
		std::endl;
		cudaFree(matrixd);                      //Free call for memory
		cudaFree(outputMatrixd);                //Free call for memory
		return false;
	}
	// Retrieve the result matrix
	for (int i=0; i<numberOfRows; i++){
		cudaMemcpy(outputMatrix[i], matrixd, numberOfColumns *sizeof(float), cudaMemcpyDeviceToHost);
		matrixd += numberOfColumns;
	}
	// Free device memory
	cudaFree(outputMatrixd);                       //Free call for memory
	cudaFree(matrixd);                             //Free call for memory
	// Success
	return true;
}
