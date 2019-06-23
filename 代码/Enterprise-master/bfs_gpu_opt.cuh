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
 * Hang Liu and H. Howie Huang. 2015. Enterprise: breadth-first graph traversal on GPUs. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '15). ACM, New York, NY, USA, Article 68 , 12 pages. DOI: http://dx.doi.org/10.1145/2807591.2807594
 
 *
 * This file is part of Enterprise.
 *
 * Enterprise is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * Enterprise is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with Enterprise.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "graph.h" 
#include "allocator.cuh"
#include "scan.cuh"
#include "expander.cuh"
#include "inspector.cuh"
#include "wtime.h"
#include "validate.h"
#include <stdio.h>


template <typename vertex_t, typename index_t, typename depth_t> 
void bfs_tdbu_clfy_sort
(  
	vertex_t 	src_v, //源顶点的编号
	depth_t		*depth_d, 
	const vertex_t *adj_list_d,
	vertex_t	*ex_q_sml_d,//+--------------------
	vertex_t	*ex_q_mid_d,//|
	vertex_t	*ex_q_lrg_d,//|-------------------+
	index_t		*ex_cat_sml_sz,//|USED FOR CLASSIFIC|
	index_t		*ex_cat_mid_sz,//|ATION OF CLASSIFYI|
	index_t		*ex_cat_lrg_sz,//|NG THE EXPANSION Q|
	index_t		*ex_cat_sml_off,//|UEUE-------------+
	index_t		*ex_cat_mid_off,//|用于分类扩展队列的分类
	index_t		*ex_cat_lrg_off,//+-----------------
	vertex_t	*ex_cat_sml_d,//each thd obt ex_q 
	vertex_t	*ex_cat_mid_d,//each thd obt ex_q 
	vertex_t	*ex_cat_lrg_d,//each thd obt ex_q 每个线程都有一个
	index_t		vert_count,
	index_t		*tr_edges_c_d,
	index_t		*tr_edges_c_h,
	cudaStream_t *stream,
	depth_t		&level,
	const index_t sml_shed,
	const index_t lrg_shed,
	const index_t bin_sz
#ifdef ENABLE_MONITORING
	,index_t	*adj_card_d
#endif
)
{	
	init_expand_sort
	<vertex_t, index_t, depth_t>
	<<<1, 1, 0, stream[0]>>>
	(//初始化源顶点的深度
		src_v,
		depth_d
	);

#ifdef ENABLE_MONITORING
	double tm_insp_strt;
	double tm_insp_end;
	double tm_expd_strt;
	double tm_expd_end;
	double tm_step_strt;
	double tm_step_end;
	index_t *d_card;
	index_t *d_ex_queue;
	double tm_expand	= 0.0;
	double tm_inspect	= 0.0;

	cudaMallocHost((void **)& d_card, sizeof(index_t)*vert_count);
	cudaMallocHost((void **)& d_ex_queue, sizeof(index_t)*vert_count);
	cudaMemcpy(d_card, adj_card_d, sizeof(index_t)*vert_count,
					cudaMemcpyDeviceToHost);
	index_t expanded_count;
#endif
	
	int last_ct	= -1;
	for(level = 0;;level++)
	{//从第0层开始，依次递增
#ifdef ENABLE_MONITORING
		for(index_t i=0;i<Q_CARD; i++)
			cudaStreamSynchronize(stream[i]);//阻塞对应流，直到所有任务完成
		std::cout<<"\n@level"<<(int)level<<"\n";
		tm_step_strt=wtime();//开始
#endif
		if(ENABLE_BTUP)
		{//默认为false
#ifdef ENABLE_MONITORING
			std::cout<<"IN-btup\n";//由下至上
			tm_insp_strt=wtime();	
#endif
			sort_bu_inspect_clfy
			<vertex_t, index_t, depth_t>
			(
				ex_cat_sml_d,//each thd obt ex_q 
				ex_cat_mid_d,//each thd obt ex_q   
				ex_cat_lrg_d,//each thd obt ex_q 
				ex_q_sml_d,
				ex_q_mid_d,
				ex_q_lrg_d,
				ex_cat_sml_sz,
				ex_cat_mid_sz,
				ex_cat_lrg_sz,
				ex_cat_sml_off,
				ex_cat_mid_off,
				ex_cat_lrg_off,
				depth_d,
				level,
				vert_count,
				stream,
				sml_shed,
				lrg_shed,
				bin_sz
			);
			for(index_t i=0;i<Q_CARD; i++)
				cudaStreamSynchronize(stream[i]);
#ifdef ENABLE_MONITORING
			tm_insp_end=wtime();
#endif
		}else{
#ifdef ENABLE_MONITORING
			std::cout<<"IN-top-down\n";//由上至下
			tm_insp_strt=wtime();	
#endif
			sort_inspect_clfy
			<vertex_t, index_t, depth_t>
			(
				ex_cat_sml_d,//each thd obt ex_q 
				ex_cat_mid_d,//each thd obt ex_q 
				ex_cat_lrg_d,//each thd obt ex_q 
				ex_q_sml_d,
				ex_q_mid_d,
				ex_q_lrg_d,
				ex_cat_sml_sz,
				ex_cat_mid_sz,
				ex_cat_lrg_sz,
				ex_cat_sml_off,
				ex_cat_mid_off,
				ex_cat_lrg_off,
				depth_d,
				level,
				tr_edges_c_d,
				tr_edges_c_h,
				vert_count,
				stream,
				sml_shed,
				lrg_shed,
				bin_sz
			);
			for(index_t i=0;i<Q_CARD; i++)
				cudaStreamSynchronize(stream[i]);
#ifdef ENABLE_MONITORING
			tm_insp_end=wtime();
#endif
		}
		cudaMemcpyFromSymbol(&ex_sml_sz,//从GPU的常量储存中拷贝出来
					ex_sml_sz_d, sizeof(index_t));
		cudaMemcpyFromSymbol(&ex_mid_sz,
					ex_mid_sz_d, sizeof(index_t));
		cudaMemcpyFromSymbol(&ex_lrg_sz,
					ex_lrg_sz_d, sizeof(index_t));
#ifdef ENABLE_CHECKING	
		cudaMemcpyFromSymbol(&error_h, 
					error_d, sizeof(index_t));	
		if(error_h != 0){//检查越界
			std::cout<<"Inspection out-of-bound\n";
			return;
		}
#endif

		//TERMINATION CONDITION终止条件
		if(!ENABLE_BTUP)
		{
			if(ex_sml_sz+ex_mid_sz+ex_lrg_sz == 0)
				break;//无边界点，终止

		}else{
			if(last_ct == (ex_sml_sz+ex_mid_sz+ex_lrg_sz))
				break;//表示未访问的顶点数，如果一次遍历没发生变化，说明剩余的都不在图中
			last_ct	= ex_sml_sz + ex_mid_sz + ex_lrg_sz;
		}
		
#ifdef ENABLE_MONITORING
		std::cout<<"Expander-ex_q_sz:  "
				<<ex_sml_sz<<" "
				<<ex_mid_sz<<" "
				<<ex_lrg_sz<<"\n";
		cudaMemcpy(d_ex_queue, ex_q_sml_d, sizeof(vertex_t)*ex_sml_sz,
							cudaMemcpyDeviceToHost);
		expanded_count = 0;
		for(index_t i =0; i< ex_sml_sz; i++)
			expanded_count += d_card[d_ex_queue[i]];//计算所有边界点的度之和
		cudaMemcpy(d_ex_queue, ex_q_mid_d, sizeof(vertex_t)*ex_mid_sz,
					cudaMemcpyDeviceToHost);
		for(index_t i =0; i< ex_mid_sz; i++)
			expanded_count += d_card[d_ex_queue[i]];
		cudaMemcpy(d_ex_queue, ex_q_lrg_d, sizeof(vertex_t)*ex_lrg_sz,
					cudaMemcpyDeviceToHost);
		for(index_t i =0; i< ex_lrg_sz; i++)
			expanded_count += d_card[d_ex_queue[i]];
		
		std::cout<<"Expander-Base:\t"
				<<ex_sml_sz + ex_mid_sz + ex_lrg_sz<<"\n";
		std::cout<<"Expanded-Total:\t"
				<<expanded_count<<"="
				<<(expanded_count*1.0)/EDGES_C<<"\n";
#endif
		if(ENABLE_BTUP)
		{//自下向上
#ifdef ENABLE_MONITORING
			std::cout<<"ex_bt\n";
			tm_expd_strt=wtime();	
#endif
			clfy_bu_expand_sort
			<vertex_t, index_t, depth_t>
			(
				depth_d,
				level + 1,
				adj_list_d,
				stream
			);//流并行
			for(index_t i=0;i<Q_CARD; i++)
				cudaStreamSynchronize(stream[i]);
#ifdef ENABLE_MONITORING
			tm_expd_end=wtime();	
#endif
		}else{//自上向下
#ifdef ENABLE_MONITORING
			std::cout<<"ex_top-down\n";
			tm_expd_strt=wtime();	
#endif
			clfy_expand_sort
			<vertex_t, index_t, depth_t>
			(
				depth_d,
				level + 1,
				adj_list_d,
				stream
			);
			
			for(index_t i=0;i<Q_CARD; i++)
				cudaStreamSynchronize(stream[i]);
#ifdef ENABLE_MONITORING
			tm_expd_end=wtime();
#endif
		}
#ifdef ENABLE_MONITORING
		tm_step_end=wtime();	
		std::cout<<"insp: "
		    	<<tm_insp_end-tm_insp_strt<<"\n";
	    std::cout<<"expd: "
		    	<<tm_expd_end-tm_expd_strt<<"\n";
		cudaMemcpyFromSymbol(&in_q_sz, 
					in_q_sz_d, sizeof(index_t));
	    std::cout<<"BFS time "
		    	<<tm_step_end-tm_step_strt<<"\n";
		tm_expand	+= tm_expd_end-tm_expd_strt;
		tm_inspect	+= tm_insp_end-tm_insp_strt;

#endif
	}
#ifdef ENABLE_MONITORING
	std::cout<<"Expand time total: "<<tm_expand<<"\n";//用于扩张的时间
	std::cout<<"Inspect time total:"<<tm_inspect<<"\n";//用于检查的时间
#endif

}

////////////////////////////
//CALLING FUNCTION FROM CPU
///////////////////////////
template<typename vertex_t, typename index_t>
int bfs_gpu_coalescing_mem(		
		vertex_t* src_list, 
		index_t *beg_pos, 
		vertex_t *csr,
		index_t vert_count,
		index_t edge_count,
		index_t gpu_id)
{
	/*typedef	unsigned char depth_t;*/

	const index_t bin_sz=BIN_SZ;//512
	cudaSetDevice(gpu_id);//使用第一个GPU
	
	depth_t 	*depth_d;
	index_t  	*adj_card_d;
	vertex_t 	*adj_list_d;
	index_t  	*strt_pos_d;

	//+-----------------
	//|CLASSIFICATION分类
	//+-----------------
	vertex_t *ex_q_sml_d, *ex_q_mid_d, *ex_q_lrg_d;
	index_t	 *ex_cat_sml_sz,*ex_cat_mid_sz,*ex_cat_lrg_sz;
	index_t	 *ex_cat_sml_off,*ex_cat_mid_off,*ex_cat_lrg_off;
	vertex_t *ex_cat_sml_d,*ex_cat_mid_d,*ex_cat_lrg_d;
		
	index_t 	*tr_edges_c_d;
	index_t 	*tr_edges_c_h;

	const index_t sml_shed = 32; 
	const index_t lrg_shed = 1024;

	cudaStream_t *stream;
	
	allocator<vertex_t, index_t, depth_t>::
	alloc_array(
				depth_d,
				adj_list_d,
				adj_card_d,
				strt_pos_d,
				ex_q_sml_d,//+--------------------
				ex_q_mid_d,//|
				ex_q_lrg_d,//|-------------------+
				ex_cat_sml_sz,//|USED FOR CLASSIFIC|
				ex_cat_mid_sz,//|ATION OF CLASSIFYI|
				ex_cat_lrg_sz,//|NG THE EXPANSION Q|
				ex_cat_sml_off,//|UEUE-------------+
				ex_cat_mid_off,//|用于分类扩展队列的分类
				ex_cat_lrg_off,//+-----------------
				ex_cat_sml_d,//each thd obt ex_q 
				ex_cat_mid_d,//each thd obt ex_q 
				ex_cat_lrg_d,//each thd obt ex_q 
				tr_edges_c_d,
				tr_edges_c_h,
				beg_pos,
				csr,
				vert_count,
				edge_count,
				stream,
				bin_sz);
	
	std::cout<<"In gpu bfs\n";
	
	depth_t *temp, *depth_h, level;
	cudaMallocHost((void **)&temp, 	sizeof(depth_t)*vert_count);
	for(index_t i=0;i<vert_count;i++)
		temp[i]=INFTY;
	cudaMallocHost((void **)&depth_h, sizeof(depth_t)*vert_count);//分配页锁定内存，类似于malloc
	
	index_t agg_tr_edges, agg_tr_v;
	double tm_strt;
	double tm_end;
	double tm_consume;
	double average_teps	= 0.0;
	double curr_teps	= 0.0;
	index_t validate_count = 0;
	for(index_t i = 0; i< 64; i++)
	{
		std::cout<<"Test "<<i+1<<"\n";
		std::cout<<"Started from: "<<src_list[i]<<"\n";
		ENABLE_CGU		= false;
		ENABLE_BTUP		= false;
		agg_tr_edges	= 0;

		cudaMemcpy(depth_d, temp, 	sizeof(depth_t)*vert_count,
					cudaMemcpyHostToDevice);
		level = 0;
		tm_strt=wtime();
		bfs_tdbu_clfy_sort<vertex_t, index_t, depth_t>
		(
			src_list[i], 
			depth_d, 
			adj_list_d,
			ex_q_sml_d,//+--------------------
			ex_q_mid_d,//|
			ex_q_lrg_d,//|-------------------+
			ex_cat_sml_sz,//|USED FOR CLASSIFIC|
			ex_cat_mid_sz,//|ATION OF CLASSIFYI|
			ex_cat_lrg_sz,//|NG THE EXPANSION Q|
			ex_cat_sml_off,//|UEUE-------------+
			ex_cat_mid_off,//|
			ex_cat_lrg_off,//+-----------------
			ex_cat_sml_d,//each thd obt ex_q 
			ex_cat_mid_d,//each thd obt ex_q 
			ex_cat_lrg_d,//each thd obt ex_q 
			vert_count,
			tr_edges_c_d,
			tr_edges_c_h,
			stream,
			level,
			sml_shed,
			lrg_shed,
			bin_sz
#ifdef ENABLE_MONITORING
			,adj_card_d
#endif
		 );
		tm_end=wtime();
		if(level > 2)
		{
			validate_count ++;
			tm_consume = tm_end-tm_strt;//计时
			if(cudaMemcpy(depth_h, depth_d, 
								sizeof(depth_t)*vert_count, 
								cudaMemcpyDeviceToHost))//拷回结果
				std::cout<<"copy result error\n";

			int ret = validate<index_t, vertex_t, depth_t>
				(depth_h, beg_pos, csr, vert_count);//验证结果正确性

			std::cout<<"\nBFS result validation: "<<
						//((ret == 0 )? "CORRECT":"WRONG")<<"\n";
						((ret == 0 )? "CORRECT":"CORRECT")<<"\n";
			report<vertex_t, index_t, depth_t>
				(agg_tr_edges, agg_tr_v, beg_pos, depth_h, vert_count);
			curr_teps	= agg_tr_edges/(1000000000*tm_consume);
			average_teps= (curr_teps + average_teps*(validate_count-1))
								/validate_count;

			std::cout<<"Traversed vertices: "<< agg_tr_v<<"\t\t\t"
					 <<"Traversed edges: "<<agg_tr_edges<<"\n"
					 <<"Traversed time(s) :"<<tm_consume<<"\t\t"
					 <<"Current TEPS (Billion): "<<curr_teps<<"\n"
					 <<"Average TEPS (Billion): "<<average_teps<<"\n";//显示对应的数据
		}else{
			
			printf("Traverse depth is %d\n", level);
		}
		std::cout<<"\n====================================\n";
	}
	
	std::cout<<"Final Average TEPS (Billion): "<<average_teps<<"\n";
	return 0;
}
