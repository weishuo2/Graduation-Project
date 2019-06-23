#/*
# * Parallel Graph Coloring: 
# * Author: Kartik Mankad 
# * Email: kmankad@ncsu.edu
# * Description: A parallel implementation of the FirstFit Graph Coloring algorithm
# */
#include <iostream>
#include <vector>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <helper_functions.h> // For the SDKTimer
#include "FirstFitCUDA.h"
#include "GraphReader.h"

using namespace std;

// Max number of iterations to run
#define MAX_NUM_ITERATIONS 10

// Declare the GraphReader Object
GraphReader *GReader;
// Declare the CSR Graph Object
CSRGraph *InputGraph;
// Declare the device side raw arrays
int *d_ColIdx, *d_RowPtr, *d_ColorVector;
bool *d_changed,*changed, *d_ColorValid;
// String vars to hold the filenames
string InputMTXFile;
string OutputTxtFile;
// Timer
float ExecTime;
StopWatchInterface *timer = NULL;

// Simple function to print usage
void PrintUsage(int argc, char* argv[]){
	cout << "Usage: " << argv[0] << " <input-mtx-file> <output-txt-file> " << endl;
	exit(2);
}

void Initialize(int argc, char* argv[]){
	//TODO: Add a more robust commandline option parsing mechanism
	if (argc != 3){
		PrintUsage(argc,argv);
	}
	// Construct the GraphReader
	GReader = new GraphReader();
	// Read the input file into a CSRGraph object
	InputGraph = GReader->ReadCSR(argv[1]);
	OutputTxtFile = argv[2];

	// Print the Input Graph
	//InputGraph->Print();

	// Allocate Device side vectors/ints
	// TODO: Enhance the CSRGraph to contain
	// the allocation for Device side vars. We'd protect that with __CUDACC__
	CUDA_CALL(cudaMalloc((void**)&d_ColIdx,sizeof(int)*(InputGraph->GetNumEdges()))); 
	CUDA_CALL(cudaMalloc((void**)&d_RowPtr,sizeof(int)*(InputGraph->GetNumVertices()+1))); 
	CUDA_CALL(cudaMalloc((void**)&d_ColorVector,sizeof(int)*(InputGraph->GetNumVertices()))); 
	CUDA_CALL(cudaMalloc((void**)&d_ColorValid,sizeof(bool)*(InputGraph->GetNumVertices()))); 
	// TODO See if the SM version is high enough to use UVM
	CUDA_CALL(cudaMemcpy(d_ColIdx, &InputGraph->ColIdx.front(), sizeof(int)*(InputGraph->GetNumEdges()), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_RowPtr, &InputGraph->RowPtr.front(), sizeof(int)*(InputGraph->GetNumVertices()+1), cudaMemcpyHostToDevice));
	//	CUDA_CALL(cudaMemcpy(d_ColorVector, &InputGraph->ColorVector.front(), sizeof(int)*(InputGraph->GetNumVertices()), cudaMemcpyHostToDevice));

	// Allocate device side memory for the changed var
	CUDA_CALL(cudaMalloc((void**)&d_changed, sizeof(bool)));	

	// Create a timer to measure execution
	sdkCreateTimer(&timer);
}

int main (int argc, char* argv[]){

	// Initialize the program
	Initialize(argc, argv);

	//TODO: Experiment with block sizes
	dim3 GridSize ((InputGraph->GetNumVertices()-1)/32 + 1, 1, 1);
	dim3 BlockSize (32, 1, 1); 
	// Call the init kernel
	InitializeColorVector<<<GridSize, BlockSize>>>(InputGraph->GetNumVertices(), d_ColorValid, d_ColorVector);		
	CUDA_CHECK();
	// TODO: Try DP and move this loop to the device
	for (int i=0; i<MAX_NUM_ITERATIONS; i++){
		changed = new bool(true);	
		while(*changed == true){
			// Start the timer
			sdkStartTimer(&timer);
			// Color the graph
			ColorGraph<<<GridSize, BlockSize>>>(InputGraph->GetNumVertices(), InputGraph->GetNumEdges(), d_ColIdx, d_RowPtr, d_ColorVector, d_changed);
			// Resolve any invalid coloring scenarios
			ResolveBadColoring<<<GridSize, BlockSize>>>(InputGraph->GetNumVertices(),d_ColIdx, d_RowPtr, d_ColorVector, d_ColorValid); 
			// Check for any errors with the kernel launch
			CUDA_CHECK();
			// Stop and get the time
			sdkStopTimer(&timer);
			// Get the changed var back to the host to see if we need to proceed
			CUDA_CALL(cudaMemcpy(changed, d_changed, sizeof(bool), cudaMemcpyDeviceToHost));
		}
		ExecTime += sdkGetTimerValue(&timer);
		// Reset the timer for the next iteration
		sdkResetTimer(&timer);
		// Wait for the kernels to finish before going for the next pass
		CUDA_CALL(cudaDeviceSynchronize());
	} // end MAX_NUM_ITERATIONS loop

	// Copy color vector back
	CUDA_CALL(cudaMemcpy(&InputGraph->ColorVector.front(),d_ColorVector, sizeof(int)*(InputGraph->GetNumVertices()), cudaMemcpyDeviceToHost));

	// Verify the coloring
	if (InputGraph->VerifyColoring() == false){
		LogError("Incorrect Coloring!");
	}

	LogInfo("Execution Time: %0.2fms", ExecTime);

	// Print the coloring to STDOUT and an output file
	//InputGraph->PrintColoring();
	InputGraph->DumpColoringToFile(OutputTxtFile);

	// Free the device side vectors
	CUDA_CALL(cudaFree(d_ColIdx));
	CUDA_CALL(cudaFree(d_RowPtr));
	CUDA_CALL(cudaFree(d_ColorVector));
	CUDA_CALL(cudaFree(d_ColorValid));

	// Delete the timer
	sdkDeleteTimer(&timer);

	// cudaDeviceReset causes the driver to clean up all state. While
	// not mandatory in normal operation, it is good practice.  It is also
	// needed to ensure correct operation when the application is being
	// profiled. Calling cudaDeviceReset causes all profile data to be
	// flushed before the application exits
	cudaDeviceReset();

}
