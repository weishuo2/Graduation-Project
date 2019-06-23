//Graph format: Json based format: [src_id, src_weigh,[[connected_ver_0, edge_weight],[connected_ver_1, edge_weight],[connected_ver_2, edge_weight]]]
//Storage format: 
//struct{
//		int: src_ver
//		Arr: [ver_0|ver_1|ver_2|...]
//		Int: num_conn_ver
//	}
#ifndef	GRAPH_H
#define	GRAPH_H

#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <queue>
#include "comm.h"
#include "graph_reader.h"

template 
<	typename vertex_t, 
	typename index_t, 
	typename depth_t>
class graph{
	
	//variable
public:
	vertex_t	*src_list;//源点数组
	index_t		src_count;//源点数
	index_t 	edge_count;//边数
	index_t 	vert_count;//顶点数
	depth_t		*depth;//深度
	vertex_t	*parent;
	
	csr_graph ggraph;
	tdata 		*gdata;
	index_t		sml_shed;
	index_t		lrg_shed;
	index_t		gpu_id;
	index_t		*gpu_ranger;
	index_t 	num_agg_bfs;
	index_t		sw_level;
	
	int world_sz, my_id;
	//constructor
public:
	graph() {};
	graph(	//构造函数
		graph_reader * g,
			index_t			sw_level,
		index_t			vert_count,
		index_t			edge_count,
		index_t			num_groups,
		index_t			num_agg_bfs,
		int world_sz, int my_id,
		index_t			gpu_id,
		index_t			sml_shed,
		index_t			lrg_shed);

	//functions
public:
	int write_result();
	int bfs_gpu_coalescing_mem();
	int alloc_array();

};

#include "bfs_gpu_opt.cuh"
#include "graph.cuh"
#include "write_result.cuh"
#include "allocator.cuh"
#endif
//included in 	main.cpp
//		bfs_seq.cpp
//		bfs.cu
