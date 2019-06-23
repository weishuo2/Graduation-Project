#ifndef _COLORING_H_
#define _COLORING_H_

#define CUDA_MAX_BLOCKS 128

void before_mix_color(
    int n_edges,
    int n_vertices,
    int * colors,
    int * srcs,
    int * dsts,
    int * col,
    int * row_ptr,
    int * row,
    int * col_ptr
);

void after_mix_color(void);

void jplColor(const int n_vertices, 
              const int n_edges, 
              const int *row_ptr, 
              const int *col, 
              const int * col_ptr, 
              const int * row, 
              int *colors) ;
void greedyColor(const int n_vertices, 
                 const int n_edges, 
                 const int *row_ptr, 
                 const int *col, 
                 const int *col_ptr, 
                 const int *row, 
                 int *colors) ;
void colorBycuSPARSE(const int n_vertices, 
                     const int n_edges, 
                     const int *row_ptr, 
                     const int *col, 
                     const int * col_ptr, 
                     const int * row, 
                     int *colors) ;
void colorByEdgeOnGPU(const int n_vertices,
                      const int n_edges, 
                      const int * srcs, 
                      const int * dsts, 
                      const int *row_ptr, 
                      const int *col, 
                      const int *col_ptr, 
                      const int *row,
                      int * colors);

// color the vertices which are targets
void greedyColor2(const int n_vertices, 
                  const int *row_ptr, 
                  const int *col, 
                  const int * col_ptr, 
                  const int * row, 
                  const int start_c, 
                  const int * is_target,
                  int *colors);

void mixColor(const int n_vertices, 
              const int n_edges, 
              const int *row_ptr, 
              const int *col, 
              const int *col_ptr, 
              const int *row,
              const int max_iter,
              const float fraction,
              int * colors,
              float fraction2);

#endif
