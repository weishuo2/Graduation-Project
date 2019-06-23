/*
 * Copyright 2016 The George Washington University
 * Written by Hang Liu 
 * Directed by Prof. Howie Huang
 *
 * https://www.seas.gwu.edu/~howie/
 * Contact: iheartgraph@gmail.com
 *
 * 
 * Please cite the following paper:
 * 
 * Hang Liu, H. Howie Huang and Yang Hu. 2016. iBFS: Concurrent Breadth-First Search on GPUs. Proceedings of the 2016 International Conference on Management of Data. ACM. 
 *
 * This file is part of iBFS.
 *
 * iBFS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * iBFS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with iBFS.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "gpu_ibfs.cuh"
#include "graph.h"
#include <sstream>
#include <iostream>
#include <fstream>

int main(int args, char *argv[]) {
	typedef long vertex_t;
	typedef long index_t;
	typedef unsigned char depth_t;
	typedef unsigned long long int comp_t;
	std::cout<<"Input: /path/to/exe /path/to/beg "
				<<"/path/to/csr concurrent-count " //并发数
				<<"switch-level bfs_count\n";
	if(args != 6){std::cout<<"Wrong input\n";exit(-1);}
	
	const char *beg_file=argv[1];
	const char *csr_file=argv[2];
	const int concurr_count=atoi(argv[3]);
	const depth_t sw_level=atoi(argv[4]);
	const index_t src_count=atoi(argv[5]);//源点数
	graph *g=new graph(beg_file, csr_file, src_count);//初始化图
	g->gen_src();//生成源点
	g->groupby();

	cudaSetDeviceFlags(cudaDeviceMapHost);//如果调用了同步版本的GPU函数，只能在设备完成任务后才能返回主机线程，此时主机端进入yield，block或者spin状态，可以由cudaSetDeviceFlags（）或cuCtxCreate（）来选择主机端在进行GPU计算时进入的状态
	const index_t sml_shed 	= 32; 
	const index_t lrg_shed 	= 512;
	index_t gpu_count	= 1;//gpu数目
	
	gpu_ibfs *ibfs_inst = new gpu_ibfs //遍历初始化
		((const graph *)g, 
			concurr_count, 
			sw_level,
			gpu_count, 
			sml_shed, 
			lrg_shed);
	

	//keep track of orphans跟踪单点
	comp_t *tmp_depth;
	H_ERR(cudaMallocHost((void **)&tmp_depth, 
			sizeof(comp_t)*g->vert_count*ibfs_inst->joint_count));
	
	std::cout<<"Start bfs ...\n";
	double tm_ibfs=wtime();
	for(index_t i=0; i<src_count/concurr_count; i++)//源点数目，并发数指一次运行几个bfs
	{//运行次数		
		index_t grp_beg=i*concurr_count;
		cudaMemcpy(ibfs_inst->depth_comp_last,tmp_depth,
			sizeof(comp_t)*g->vert_count*ibfs_inst->joint_count, 
			cudaMemcpyHostToDevice);
		cudaMemcpy(ibfs_inst->depth_comp_curr,tmp_depth,
			sizeof(comp_t)*g->vert_count*ibfs_inst->joint_count, 
			cudaMemcpyHostToDevice);
	
		double tm_ibfs=wtime();
		ibfs_inst->traversal(grp_beg);//开始运行算法
		std::cout<<"Traversal-time: "<<wtime()-tm_ibfs<<" second(s)\n";
		//graph_d->write_result();
	}
	std::cout<<"Total time: "<<wtime()-tm_ibfs<<"\n";
	return 0;
}
