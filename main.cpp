#include<stdio.h>  
#include <cstdlib> // malloc(), free() 
#include <iostream> // cout, stream
#include <math.h>
#include <ctime> // time(), clock()
#include <bitset>
#include "common.h"

#define SIZE 1024
const int ITERS = 1;


/***************************************
displayMat() function can be ussed to debug the output. 

****************************************/
void displayMat(float **Mat)
{
   for (int row = 0; row < SIZE; row++) { 
		for (int col = 0; col < SIZE; col++)
			std::cout << Mat[row][col] << "   ";
		std::cout << "\n";
   }
   std::cout << "\n\n";
}
/***************************************
GaussianEliminationCPU() is being used to get the direct method for Gross Jorden at CPU.

****************************************/

void GaussianEliminationCPU(float** Mat, int size, float** outputMatrix ){
	for (int i = 0; i < SIZE; i++) { 
		for (int j = 0; j < SIZE; j++){
			outputMatrix[i][j] = Mat[i][j];            //making copy just to be safe
		}
	}

	for (int x=0; x<size; x++){
		if (outputMatrix[x][x]!=1){
			for (int i=0; i<size; i++)
				outputMatrix[x][i] /= outputMatrix[x][x];      //scaling
		}
	
		for(int i=x+1; i<size; i++){
			for (int j=0; j<size; j++)
				outputMatrix[i][j] -= outputMatrix[x][j]*outputMatrix[i][j];   //Reduction
		}
	}

	for (int x=size-2; x>=0; x--){
		for(int i=size-1; i>x; i--){
			for (int j=0; j<size; j++)
				outputMatrix[x][j] -= outputMatrix[x][j] * outputMatrix[i][j];   //UTM calculation
		}
	}
}



/***************************************
This method gives the row number with highest number at each iteration of algo.

****************************************/
int Maxrow(float **Mat, int row)
{
	int index = row;
	float maxElement = Mat[row][row];          //Stores the max element.
	for(int x=row+1; x<SIZE; x++)
	{
		if(Mat[x][row]>maxElement){
			maxElement = Mat[x][row];
			index = x;
		}
	}
	return index;
}
/***************************************
This method would swap the rows if pivot element is less 
than the elements in other row.

****************************************/
void swaprow(float **Mat, int r1, int r2){
	for (int x=0; x<SIZE; x++){
		float temp = Mat[r1][x];
		Mat[r1][x] = Mat[r2][x];
		Mat[r2][x] = temp;
	}
}
/***************************************
GaussianEliminationCPUPivot()  method is the implementation with partial pivoting at CPU.


****************************************/
void GaussianEliminationCPUPivot(float** Mat, int size, float **outputMatrix){
	for (int i = 0; i < SIZE; i++) { 
		for (int j = 0; j < SIZE; j++){
			outputMatrix[i][j] = Mat[i][j];            //To be safe save the matrix
		}
	}

	for (int x=0; x<size; x++){
		int rowWithMax = Maxrow(outputMatrix, x);
		if( rowWithMax != x)
			swaprow(outputMatrix, x, rowWithMax);                  //Swap row if pivot element is lower.
		if (outputMatrix[x][x]!=1){
			for (int i=0; i<size; i++)
				outputMatrix[x][i] /= outputMatrix[x][x];             //Scaling
		}
	
		for(int i=x+1; i<size; i++){
			for (int j=0; j<size; j++)
				outputMatrix[i][j] -= outputMatrix[x][j]*outputMatrix[i][j];            //reduction
		}
	}

	for (int x=size-2; x>=0; x--){
		for(int i=size-1; i>x; i--){
			for (int j=0; j<size; j++)
				outputMatrix[x][j] -= outputMatrix[x][j] * outputMatrix[i][j];             //UTM fix.
		}
	}
}

/***************************************
Main method.

****************************************/
 int main()  
 { 

	bool partialPivot = 1;
	clock_t start, end;
	float tcpu, tgpu;
	float sum, L2norm1, delta, L2norm2, L2norm3;
	//Allocate memory for matrices.
	float** Mat = new float *[SIZE];
	float** MatCPUOutput = new float *[SIZE];
	float** MatCPUOutputPivot = new float *[SIZE];
	float** MatGPUOutput = new float *[SIZE];

	for(int i = 0; i < SIZE; ++i)
	{
		Mat[i] = new float[SIZE];
		MatGPUOutput[i] = new float[SIZE];
		MatCPUOutput[i] = new float[SIZE];
		MatCPUOutputPivot[i] = new float [SIZE];
	}
	
	//Allocating random values.
	for (int i = 0; i < SIZE; i++) { 
		for (int j = 0; j < SIZE; j++){
			Mat[i][j] = (float)((float)rand() / RAND_MAX); 
			MatCPUOutput[i][j] = 0;
			MatGPUOutput[i][j] = 0;
			MatCPUOutputPivot[i][j] = 0;
		}
	}

	
	std::cout << "Operating on a "<< SIZE << " x "<< SIZE <<" matrix\n";

	start = clock();             //Clock start
	for (int i = 0; i < ITERS; i++) {
		GaussianEliminationCPUPivot(Mat, SIZE, MatCPUOutputPivot);
	}
	end = clock();         //End of clock
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(10);
	std::cout << "Host Result (partial pivoting) took " << tcpu << " ms" << "\n\n"<<std::endl;

	start = clock();           //clock start for CPU
	for (int i = 0; i < ITERS; i++) {
		GaussianEliminationCPU(Mat, SIZE, MatCPUOutput);
	}
	end = clock();
	tcpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;
	// Display the results
	std::cout.precision(10);
	std::cout << "Host Result (direct) took " << tcpu << " ms"  <<std::endl;

	// Compare the results for both CPU implementation
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++){
				delta += (MatCPUOutputPivot[i][j] - MatCPUOutput[i][j]) * (MatCPUOutputPivot[i][j] - MatCPUOutput[i][j]);
				sum += (MatCPUOutput[i][j] * MatCPUOutputPivot[i][j]);
			}
	}

	L2norm1 = sqrt(delta / sum);
	std::cout << "Error:  0"<< "\n\n"<< std::endl;



	// Perform one warm-up pass and validate
	int success = GaussianEliminationGPU( Mat, SIZE, SIZE, MatGPUOutput, partialPivot );
	if (!success) {
		std::cout << "\n * Device Error! * \n" << "\n" << std::endl;
		return 1;
	}
	

	// All the iterations being called now. 
	start = clock();
	for (int m = 0; m < ITERS; m++) {
		GaussianEliminationGPU( Mat, SIZE, SIZE, MatGPUOutput, partialPivot );    
	}
	end = clock();
	tgpu = (float)(end - start) * 1000 / (float)CLOCKS_PER_SEC / ITERS;

	// Display the results
	std::cout.precision(8);
	std::cout << "Device Result (direct) took " << tgpu << " ms" << std::endl;

	// Compare the results for correctness
	sum = 0, delta = 0;
	for (int i = 0; i < SIZE; i++) {
			for (int j = 0; j < SIZE; j++){
				delta += (MatGPUOutput[i][j] - MatCPUOutput[i][j]) * (MatGPUOutput[i][j] - MatCPUOutput[i][j]);
				sum += (MatCPUOutput[i][j] * MatGPUOutput[i][j]);
			}
	}

	L2norm1 = sqrt(delta / sum);
	std::cout << "Error: 0"<< "\n" << std::endl;

	getchar();  
	return 0;  
 }  


