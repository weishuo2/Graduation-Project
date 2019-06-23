#include <iostream>
#include <cassert>
#include <algorithm>
#include <cuda_runtime.h>
#include <thrust/count.h>
#include <stdlib.h>
#include <string.h>
#include "input_data.h"
#include "types.h"
#include "coloring.h"
#include "utility.h"
#include "time.h"
using namespace std;

bool start_index_is_zero;

int getNumColors(int n, const int * colors)
{
    bool * colors_used = (bool *)malloc(sizeof(bool) * n);
    memset(colors_used, 0, sizeof(bool) * n);
    int num_colors = 0;
    for (int i = 0; i < n; ++i)
        colors_used[colors[i]] = true;
    num_colors = std::count(colors_used, colors_used + n, true);
    return num_colors;
}

bool isRight(int n, int * row_ptr, int * col, int * col_ptr, int * row, int * colors)
{
  // check if all vertices are colored
  for (int i = 0; i < n; ++i)
  {
    if (colors[i] == -1)
    {
      //cout << "Not all vertices are colored.\n";
      cout << "No " << "\t";
      return false;
    }
  }
  //cout << "All vertices are colored.\n";
  cout << "yes" << "\t";

  for (int i = 0; i < n; ++i)
  {
      int ic = colors[i];
      for (int j = row_ptr[i]; j < row_ptr[i+1]; ++j)
      {
          if (colors[col[j]] == ic && col[j] != i)
          {
              cout << "Wrong solution!\n";
              cout << i << "->" << col[j] << " color=" << ic << endl;
              return false;
          }
      }

      for (int j = col_ptr[i]; j < col_ptr[i+1]; ++j)
      {
          if (colors[row[j]] == ic && row[j] != i)
          {
              cout << "Wrong solution!\n";
              cout << row[j] << "->" << i << " color=" << ic << endl;
              return false;
          }
      }
  }
  return true;
}

// test function declaration
//void test(const char* filename, const int start_index)
void test(const char* filename, const int start_index, const int max_iters, const float fraction, int deviceNum, int iter2)
{
    degree_t * degree = NULL;
    vertex_t ** adj_list = NULL;
    int num_vertices = 0;
    int num_edges = 0;
    start_index_is_zero = (start_index == 0? true: false);
    readData(filename, &degree, &adj_list, &num_vertices, &num_edges);
    //readData((char*)"../data/test.txt", &degree, &adj_list, &num_vertices, &num_edges);
    vertex_t * row_ptr = (vertex_t*)malloc(sizeof(vertex_t) * (num_vertices + 1));
    vertex_t * col = (vertex_t*)malloc(sizeof(vertex_t) * num_edges);
    vertex_t * col_ptr = (vertex_t*)malloc(sizeof(vertex_t) * (num_vertices + 1));
    vertex_t * row = (vertex_t*)malloc(sizeof(vertex_t) * num_edges);
    consCSR(num_vertices, degree, adj_list, row_ptr, col);
    consCSC(num_vertices, degree, adj_list, col_ptr, row);

    vertex_t * dsts_of_edges = col;
    vertex_t * srcs_of_edges = (vertex_t*)malloc(sizeof(vertex_t) * num_edges);
    consSrcsOfEdges(num_vertices, degree, srcs_of_edges);

    int * colors = (int *)malloc(sizeof(int) * num_vertices);
    memset(colors, 0, sizeof(int) * num_vertices);

    struct timeval start_time, end_time;

    // print info
    /**********************
    cout << "File: " << filename << ", " << "num_vertices=" << num_vertices
         << ", num_edges=" << num_edges << ", ";
    ******************/

    // print num of connected components
    int * is_target = (int *)malloc(sizeof(int) * num_vertices);
    thrust::fill(is_target, is_target + num_vertices, 1); // init colors to -1，迭代器赋值
    //cout << "num_cc = " << getNumCC(num_vertices, row_ptr, col, col_ptr, row, is_target) << endl;
    free(is_target);

    /****************
    cout << "Running greedy coloring algorithm on CPU...\n";
    gettimeofday(&start_time, NULL);
    greedyColor(num_vertices, num_edges, row_ptr, col, col_ptr, row, colors);
    gettimeofday(&end_time, NULL);
    cout << "CPU: Greeddy coloring time: " << elapsed(start_time, end_time) << "ms" << endl
         << "Solution: num_colors=" << getNumColors(num_vertices, colors) << ", "
         << (isRight(num_vertices, row_ptr, col, col_ptr, row, colors) ? "right solution" : "wrong solution") << endl << endl;
    ********************/
    // choose device & initialize cuda
    cudaSetDevice(deviceNum);

//    cout << "Running jpl coloring algorithm on CPU...\n";
//    gettimeofday(&start_time, NULL);
//    jplColor(num_vertices, num_edges, row_ptr, col, col_ptr, row, colors);
//    gettimeofday(&end_time, NULL);
//    cout << "GPU: jpl coloring time: " << elapsed(start_time, end_time) << "ms" << endl
//         << "Solution: num_colors=" << getNumColors(num_vertices, colors) << ", "
//         << (isRight(num_vertices, row_ptr, col, col_ptr, row, colors) ? "right solution" : "wrong solution") << endl << endl;

   // cout << "Running mix coloring algorithm (Run " << max_iters << "iters on GPU)...\n";

    //std::cout << "frac" << "\t" << "Iter" << "\t" << "Trav" << "\t" << "isColored" << "\t"<< "Total" << "\t" << "isRight" <<endl;
    std::cout << fraction << "\t";
    before_mix_color(num_edges,num_vertices,colors,srcs_of_edges,dsts_of_edges,col,row_ptr,row,col_ptr);
    gettimeofday(&start_time, NULL);
    mixColor(num_vertices, num_edges, row_ptr, col, col_ptr, row, max_iters, fraction, colors,iter2);
    gettimeofday(&end_time, NULL);
    after_mix_color();
    std::cout << elapsed(start_time, end_time) 
              << "\t" << (isRight(num_vertices, row_ptr, col, col_ptr, row, colors) ? "right" : "wrong") << "\t" << getNumColors(num_vertices, colors) << endl;

/****************
    cout << "total time: " << elapsed(start_time, end_time) << "ms" << endl
         << "Solution: num_colors=" << getNumColors(num_vertices, colors) << ", "
         << (isRight(num_vertices, row_ptr, col, col_ptr, row, colors) ? "right solution" : "wrong solution") << endl << endl;
*****************/
    //cout << "Running cuSPARSE coloring algorithm ...\n";
    //gettimeofday(&start_time, NULL);
    //colorBycuSPARSE(num_vertices, num_edges, row_ptr, col, col_ptr, row, colors);
    //gettimeofday(&end_time, NULL);
    //cout << "GPU: coloring time: " << elapsed(start_time, end_time) << "ms" << endl
    //     << "Solution: num_colors=" << getNumColors(num_vertices, colors) << ", "
    //     << (isRight(num_vertices, row_ptr, col, col_ptr, row, colors) ? "right solution" : "wrong solution") << endl;

    free(colors);
    free(row_ptr);
    free(col);
    free(col_ptr);
    free(row);
    free(srcs_of_edges);
    releaseData(num_vertices, degree, adj_list);
}

/*******************************************************
int main(int argc, char ** argv)
{
    if (argc < 3)
    {
	cout << "Usage: " << argv[0] << " --file filename [--start_index number] [--max_iters max_ters] [--fraction fraction_value]" << endl;
        return 0;
    }
    string filename = "../data/web-Stanford.txt";
    int start_index = 0;
    int max_iters = 0;
    float fraction = 0;
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--file") == 0)
            filename = argv[i+1];
        if (strcmp(argv[i], "--start_index") == 0)
            start_index = atoi(argv[i+1]);
        if (strcmp(argv[i], "--max_iters") == 0)
            max_iters = atoi(argv[i+1]);
        if (strcmp(argv[i], "--fraction") == 0)
            fraction = atof(argv[i+1]);
    }
    test(filename.c_str(), start_index, max_iters, fraction);
    return 0;
}
**********************************************************/

int main(int argc, char ** argv)
{

    int start_index = 0;
    int max_iters = 0;
    float fraction = 0;
    int deviceNum = 0;
    int iter2 = 1;
    if (argc < 3)
    {
      cout << "Usage: " << argv[0] << " --file filename [--start_index number] [--max_iters max_ters] [--frac fraction_value] --dev deviceNum" << endl;
        return 0;
    }
    string filename = "../data/web-Stanford.txt";

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--file") == 0)
            filename = argv[i+1];       
        if (strcmp(argv[i], "--start_index") == 0)
            start_index = atoi(argv[i+1]);
        if (strcmp(argv[i], "--max_iters") == 0)
            max_iters = atoi(argv[i+1]);  
        if (strcmp(argv[i], "--frac") == 0)
            fraction = atof(argv[i+1]); 
        if (strcmp(argv[i], "--dev") == 0)    
            deviceNum = atof(argv[i+1]); 
        if (strcmp(argv[i], "--iter2") == 0)    
            iter2 = atof(argv[i+1]); 
    }

    std::cout << "frac" << "\t" << "Iter" << "\t" << "Trav" << "\t" << "isColored" << "\t"<< "Total" << "\t" << "isRight" << "\t"<< "Colors" << endl;
    
    //test(filename.c_str(), start_index, max_iters, fraction, deviceNum);

    /***********************************************************************
    for (fraction = 0.4; fraction <= 0.9; fraction+=0.05)
      test(filename.c_str(), start_index, max_iters, fraction, deviceNum);
    ************************************************************************/

    for (int i =0; i < 10; i++)
      test(filename.c_str(), start_index, max_iters, fraction, deviceNum, iter2);
    //C语言没有字符串，所以用这个变成字符串数组
    //test(filename.c_str(), start_index, max_iters, 0.05);
    return 0;
}