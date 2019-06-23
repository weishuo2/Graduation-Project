#ifndef _INPUT_DATA_
#define _INPUT_DATA_

#include "types.h"

void readData(const char * filename, degree_t ** degree_p, vertex_t *** adj_list_p, int * num_vertices, int * num_edges);
void consCSR(int num_vertices, degree_t * degree, vertex_t ** adj_list, vertex_t * row_ptr, vertex_t * col);
void consCSC(int num_vertices, degree_t * out_degree, vertex_t ** adj_list, vertex_t * col_ptr, vertex_t * row);
void consSrcsOfEdges(int num_vertices, int * degree, int * srcs_of_edges);
void releaseData(int num_vertices, degree_t * degree, vertex_t ** adj_list);//最后释放申请的内存空间

#endif
