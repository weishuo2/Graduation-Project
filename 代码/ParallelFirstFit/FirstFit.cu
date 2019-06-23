/*
# * Parallel Graph Coloring: 
# * Author: Kartik Mankad 
# * Email: kmankad@ncsu.edu
# * Description: A parallel implementation of the FirstFit algorithm
# */
#include "FirstFitCUDA.h"

// Init the ColorValid array
// TODO: Replace with one thrust::fill call
__global__ void InitializeColorVector(int d_NumVertices, bool* d_ColorValid, int* d_ColorVector){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID < d_NumVertices){
		d_ColorValid[threadID] = false;	
		d_ColorVector[threadID] = NO_COLOR;
	}
}//初始化

// Actual Graph Coloring kernel
__global__ void ColorGraph(int d_NumVertices, int d_NNZ, int* d_ColIdx, int* d_RowPtr, int* d_ColorVector, bool* d_changed){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	// Temp storage to store the neighbors' colors. 
	int NeighborColors[MAX_DEGREE];
	// Set the default value of changed to false
	*d_changed = false;
	if (threadID < d_NumVertices) { 
		// So that we dont walk over the edge of the d_RowPtr array
		if (d_ColorVector[threadID] == NO_COLOR){
			// if the vertex is not colored
			// Iterate over its neighbors
			int NumNeighbors = 0;
			for (int CurrNodeOffset=d_RowPtr[threadID]; CurrNodeOffset<d_RowPtr[threadID+1] ; CurrNodeOffset++){
				// Mark the neighbor's colors unavailable by
				// pushing them into the NeighborColors vector
				int NodeIndex = d_ColIdx[CurrNodeOffset];
				int NodeColor = d_ColorVector[NodeIndex];
				NeighborColors[NumNeighbors++] = NodeColor;
			}
			// Here, we have the neighbor's colors
			// as first NumNeighbors elements of the NeighborColors array
			// We go over that array to find the first possible color we can assign
			// Now that we know what colors _cant_ be used, 
			// lets find the first color that fits
			bool VertexColored = false;
			int VertexColor = 1; // We start our attempt from Color#1
			bool IsNeighborColor;
			while(VertexColored != true){
				IsNeighborColor = false;
				// Check if the color we're attempting to assign
				// is available
				for (int Neighbor=0; Neighbor < NumNeighbors; Neighbor++){
					if (NeighborColors[Neighbor] == VertexColor){
						IsNeighborColor = true;
						break;
					}
				}

				// If the color we're attempting is not already
				// assigned to one of the neighbors...
				if (IsNeighborColor == false){
					// This is a valid color to assign
					d_ColorVector[threadID] = VertexColor;
					// Indicate that we colored a vertex, so the graph state has changed
					*d_changed = true;
					// Set the VertexColored flag and break out of the while loop
					VertexColored = true;
					break;
				} else {
					// Try with the next color
					VertexColor++;
				}
			} // end of while(VertexColored !=true)
		} // end if d_ColorVector[threadID] == NO_COLOR
	} // end if (threadID < d_NNZ)
}

__global__ void ResolveBadColoring(int d_NumVertices, int* d_ColIdx, int* d_RowPtr, int* d_ColorVector, bool* d_ColorValid){
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	bool ColorValid = true;
	if ((threadID < d_NumVertices) && (d_ColorValid[threadID]==false)){
		// Iterate over the neighbors and check if the coloring is valid	
		for (int CurrNodeOffset=d_RowPtr[threadID]; CurrNodeOffset<d_RowPtr[threadID+1] ; CurrNodeOffset++){
			int NeighborColor = d_ColorVector[d_ColIdx[CurrNodeOffset]];
			if ((NeighborColor == d_ColorVector[threadID]) && (threadID<d_ColIdx[CurrNodeOffset])){
				// If the color matches with any one neighbor
				// its not valid, and we must recolor
				ColorValid=false;
				d_ColorVector[threadID] = NO_COLOR;
				break;
			} // if (NeighborColor == d_ColorVector...
		} // end of for loop that goes over neighbors

		// Update the vertex's coloring status
		d_ColorValid[threadID] = ColorValid;

	}// end of if ((threadID < d_NumVertices) && (d_ColorValid[threadID]==false)){
}
