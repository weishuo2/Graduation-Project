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
#include "comm.h"

////////////////////////////
//EXTERNAL VARIABLES外部变量
///////////////////////////
//__device__ int expand_type_d;


//////////////////////////////////
//This file is mainly about expander
//--------------------------------------------------
//	For expander, it should expand based on ex_queue_d
//and put all the expanded data into inspection对于扩展器，它应该基于ex_queue_d进行扩展，并将所有扩展数据放入检查中
//--------------------------------------------------


__device__ void __sync_warp(int predicate)
{
	while((!__all(predicate)))
	{
		;
	}
}

//This kernel is executed by one thread
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void init_expand_sort
(	
	vertex_t src_v,
	depth_t	*depth_d
)
{//初始化
	depth_d[src_v]	= 0;	
	in_q_sz_d		= 1;
	error_d 		= 0;
	return ;
}


//+---------------------------
//|for ex_q_sml_d expansion
//+---------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void THD_expand_sort
(	
	depth_t	*depth_d,
	index_t	curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_sml_sz_d;
	const depth_t LEVEL		= curr_level;
	
	const index_t GRNLTY	= blockDim.x * gridDim.x;
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	__shared__ index_t	hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	
	while(cache_ptr < HUB_SZ)
	{//初始化cache
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;

		cache_ptr += blockDim.x;
	}

	__syncthreads();

	//prefetching
	if (tid < q_sz)
	{
		ex_ver			= tex1Dfetch(tex_sml_exq, tid);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(tid<q_sz)
	{
		tid 	  	   += GRNLTY;
		if(tid < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_sml_exq, tid);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}

		index_t lane = strt_pos_curr;
		card_curr	+= strt_pos_curr;

		aq_ver_curr	= adj_list_d[lane];
		cache_ptr	= aq_ver_curr & (HUB_SZ - 1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{//查看是否在cache中
			adj_depth_curr = LEVEL - 1; 
			hub_depth[cache_ptr] = LEVEL;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		while(lane < card_curr)
		{
			lane++; 
			if(lane < card_curr)
			{//读取下一个邻接点信息
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;//改写
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();
	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{//有点迷
		//hub_depth should be 计算，调查 before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

//+------------------------------
//|ex_q_mid_d expansion
//+------------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void WAP_expand_sort
(	
	depth_t		*depth_d,
	index_t		curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_mid_sz_d;
	const depth_t LEVEL		= curr_level;

	const index_t vec_sz	= ((THDS_NUM>=32)? 32:1);
	const index_t tid		= threadIdx.x+blockIdx.x*blockDim.x;
	const index_t lane_s	= tid & (vec_sz-1);
	const index_t GRNLTY	= (blockDim.x * gridDim.x)/vec_sz;
	index_t	vec_id			= tid/vec_sz;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	
	__shared__ index_t		hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];

	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (vec_id < q_sz)
	{
		ex_ver			= tex1Dfetch(tex_mid_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(vec_id < q_sz)
	{
		vec_id 	  	   += GRNLTY;
		if(vec_id < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_mid_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}
	
		index_t lane	= lane_s + strt_pos_curr;
		aq_ver_curr		= adj_list_d[lane];

		cache_ptr		= aq_ver_curr & (HUB_SZ -1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			adj_depth_curr = LEVEL - 1; 
			hub_depth[cache_ptr] = LEVEL;
		}
		else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		card_curr	   += strt_pos_curr;
		
		while(lane < card_curr)
		{
			lane		+= vec_sz; 
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		__sync_warp(1);
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();

	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{
		//hub_depth should be investigated before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void CTA_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t	q_sz = ex_lrg_sz_d;
	index_t	vec_id	= blockIdx.x;
	const depth_t LEVEL		= curr_level;

	//used for prefetching
	vertex_t 	ex_ver;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t		adj_depth_curr, adj_depth_next;
	
	__shared__ index_t	hub_cache[HUB_SZ];
	__shared__ depth_t	hub_depth[HUB_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		hub_depth[cache_ptr]	= INFTY;
		cache_ptr += blockDim.x;
	}
	__syncthreads();

	//prefetching
	if (vec_id<q_sz)
	{
		ex_ver			= tex1Dfetch(tex_lrg_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver);
	}

	while(vec_id<q_sz)
	{
		vec_id += gridDim.x;
		
		if(vec_id < q_sz)
		{
			ex_ver			= tex1Dfetch(tex_lrg_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver);
		}
		
		index_t lane	= threadIdx.x + strt_pos_curr;

		aq_ver_curr	= adj_list_d[lane];
		cache_ptr	= aq_ver_curr & (HUB_SZ - 1);
		if(aq_ver_curr == hub_cache[cache_ptr])
		{
			adj_depth_curr 			= LEVEL - 1; 
			hub_depth[cache_ptr] 	= LEVEL;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		card_curr	   += strt_pos_curr;
		while(lane < card_curr)
		{
			//-reimburse lane for prefetch
			//-check lane and prefetch for next
			//	iteration
			lane	+= blockDim.x;
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr	= aq_ver_next & (HUB_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{
					adj_depth_next = LEVEL - 1; 
					hub_depth[cache_ptr] = LEVEL;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == INFTY)
				depth_d[aq_ver_curr]= LEVEL;
			
			aq_ver_curr	= aq_ver_next;
			adj_depth_curr= adj_depth_next;
		}
		__syncthreads();
		
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;
	}
	__syncthreads();
	
	cache_ptr	= threadIdx.x;

	while(cache_ptr < HUB_SZ)
	{
		//hub_depth should be investigated before depth_d[hub_cache[]]
		//Reason: hub_cache[] maybe blank which leads to out-of-bound 
		//			depth_d transaction
		if((hub_depth[cache_ptr] == LEVEL)
			&& (depth_d[hub_cache[cache_ptr]] == INFTY))
			depth_d[hub_cache[cache_ptr]]= LEVEL;

		cache_ptr += blockDim.x;
	}
}

//+-------------------------------------------------------------------
//|BOTTOM UP EXPANSION FUNCTIONS自下向上的扩展函数
//+--------------------------------------------------------------------
//+---------------------------
//|for ex_q_sml_d expansion
//+---------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void THD_bu_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t q_sz		= ex_sml_sz_d;
	
	const index_t GRNLTY	= blockDim.x * gridDim.x;//总线程数
	index_t tid				= threadIdx.x+blockIdx.x*blockDim.x;//线程id
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	//used for prefetching用于预取
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;
	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (tid < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_sml_exq, tid);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);//在CSR中的开始位置
	}

	while(tid<q_sz)
	{
		tid 	  	   += GRNLTY;
		if(tid < q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_sml_exq, tid);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
		}

		index_t lane = strt_pos_curr;
		card_curr	+= strt_pos_curr;//在CSR中的终止位置

		aq_ver_curr	= adj_list_d[lane];//第一个邻接点
		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);//哈希表中可能的位置
		if(aq_ver_curr == hub_cache[cache_ptr])
		{//如果他的第一个邻接点被缓存了，说明他在其下一层
			depth_d[ex_ver_curr] = LEVEL;
				
			ex_ver_curr		= ex_ver_next;
			card_curr 		= card_next;
			strt_pos_curr	= strt_pos_next;
			continue;
		}else{//没有被缓存就查看其深度
			adj_depth_curr	= depth_d[aq_ver_curr];
		}

		while(lane < card_curr)
		{
			lane++;
			if(lane < card_curr)
			{
				aq_ver_next	= adj_list_d[lane];
				
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
				if(aq_ver_next == hub_cache[cache_ptr])
				{//看下一个邻接点是否被缓存
					depth_d[ex_ver_curr] = LEVEL;
					break;
				}else{
					adj_depth_next	= depth_d[aq_ver_next];
				}
			}
			
			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			if(adj_depth_curr == LST_LEVEL)
			{//第一个邻接点是上一层的
				depth_d[ex_ver_curr] = LEVEL;
				break;
			}
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;//查找下一个邻接点
		}
		
		ex_ver_curr		= ex_ver_next;
		card_curr 		= card_next;
		strt_pos_curr	= strt_pos_next;//查找下一个边界点
	}
}

//+------------------------------
//|ex_q_mid_d expansion
//+------------------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void WAP_bu_expand_sort
(	
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)

{//粗粒度，32个线程跑一个顶点
	const index_t q_sz		= ex_mid_sz_d;

	const index_t vec_sz	= ((THDS_NUM>=32)? 32:1);//vec_sz个线程处理同一个顶点
	const index_t tid		= threadIdx.x+blockIdx.x*blockDim.x;//线程id
	const index_t lane_s	= tid & (vec_sz-1);//32个线程同时处理，保证32以下各不相同
	const index_t GRNLTY	= (blockDim.x * gridDim.x)/vec_sz;//顶点编号递增的步伐
	index_t	vec_id			= tid/vec_sz;//该线程需要处理的顶点编号
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	//used for prefetching
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	card_curr_revised, card_next_revised;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t 	adj_depth_curr, adj_depth_next;

	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{//读取中心块数据到cache
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching预取
	if (vec_id < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_mid_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);
		if(card_curr%vec_sz)
		{//需要将出度分成32分，不足32补全为32
			card_curr_revised = (((card_curr>>5)+1)<<5);
		}else{
			card_curr_revised = card_curr;
		}
	}

	while(vec_id < q_sz)
	{
		vec_id 	  	   += GRNLTY;
		if(vec_id < q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_mid_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
			if(card_next%vec_sz)
			{//补全出度数
				card_next_revised = (((card_next>>5)+1)<<5);
			}else{
				card_next_revised = card_next;
			}
		}
	
		index_t lane		= lane_s + strt_pos_curr;
		card_curr			+= strt_pos_curr;
		card_curr_revised	+= strt_pos_curr;

		//cardinality of all vertices in wap_queue 基数
		//is larger than 32
		aq_ver_curr		= adj_list_d[lane];//具体计算该线程处理的是哪一个邻接点

		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);
		__sync_warp(1);//处理同一个顶点的线程同步？
		if(__any(aq_ver_curr== hub_cache[cache_ptr]))
		{
			if(!lane_s){//保证只写一次，让第一个线程去写
				depth_d[ex_ver_curr] = LEVEL;
			}
			__sync_warp(1);
			
			ex_ver_curr		= ex_ver_next;
			card_curr 		= card_next;
			card_curr_revised= card_next_revised;
			strt_pos_curr	= strt_pos_next;
				
			continue;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}

		while(lane < card_curr_revised)
		{
			//prefetching for the next iteration预取
			lane		+= vec_sz; 
			if(lane < card_curr){//0~33 vertices, second round 32, 33
				aq_ver_next	= adj_list_d[lane];
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
			}else{//0~33 vertices, second round 34, ..., 63 vertices
				aq_ver_next	= V_NON_INC;
				cache_ptr = 0;
			}
			
			if(__any(aq_ver_next == hub_cache[cache_ptr]))
			{
				if(!lane_s){
					SET_VIS(depth_d[ex_ver_curr]);
					depth_d[ex_ver_curr] = LEVEL;
				}
				break;
			}else{
				if(lane < card_curr) adj_depth_next 
										= depth_d[aq_ver_next];
				else	adj_depth_next	= 0;
			}

			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			__sync_warp(1);
			
			if(__any(adj_depth_curr == LST_LEVEL))
			{
				if(!lane_s){
					depth_d[ex_ver_curr] = LEVEL;
				}
				
				break;
			}
			
			aq_ver_curr		= aq_ver_next;
			adj_depth_curr	= adj_depth_next;
		}
		__sync_warp(1);
		
		ex_ver_curr		= ex_ver_next;
		card_curr 		= card_next;
		card_curr_revised= card_next_revised;
		strt_pos_curr	= strt_pos_next;
	}
}

template<typename vertex_t, 
		typename index_t,
		typename depth_t>
__global__ void CTA_bu_expand_sort
(	//生成自下向上的新的边界点
	depth_t			*depth_d,
	index_t			curr_level,
	const vertex_t* __restrict__ adj_list_d
)
{
	const index_t	q_sz = ex_lrg_sz_d;
	
	index_t	vec_id 		= blockIdx.x;//直接一个block运行一个顶点
	vertex_t 	ex_ver_curr, ex_ver_next;
	index_t 	card_curr, card_next;
	index_t 	card_curr_revised, card_next_revised;
	index_t 	strt_pos_curr, strt_pos_next;
	vertex_t 	aq_ver_curr, aq_ver_next;
	depth_t		adj_depth_curr, adj_depth_next;
	const depth_t LEVEL		= curr_level;
	const depth_t LST_LEVEL	= LEVEL - 1;

	__shared__ index_t	hub_cache[HUB_BU_SZ];
	
	index_t	 cache_ptr	= threadIdx.x;
	while(cache_ptr < HUB_BU_SZ)
	{//正好用其线程在block的序号作为指示
		hub_cache[cache_ptr] 	= hub_vert[cache_ptr];
		cache_ptr += blockDim.x;
	}
	__syncthreads();
	
	//prefetching
	if (vec_id < q_sz)
	{
		ex_ver_curr		= tex1Dfetch(tex_lrg_exq, vec_id);
		card_curr		= tex1Dfetch(tex_card, ex_ver_curr);
		strt_pos_curr	= tex1Dfetch(tex_strt, ex_ver_curr);
		if(card_curr%blockDim.x)
		{//补全
			card_curr_revised = card_curr + blockDim.x 
								- (card_curr%blockDim.x); 
		}else{
			card_curr_revised = card_curr;
		}
	}

	while(vec_id < q_sz)
	{
		vec_id += gridDim.x;//处理的下一个顶点
		
		if(vec_id<q_sz)
		{
			ex_ver_next		= tex1Dfetch(tex_lrg_exq, vec_id);
			card_next		= tex1Dfetch(tex_card, ex_ver_next);
			strt_pos_next	= tex1Dfetch(tex_strt, ex_ver_next);
			if(card_next%blockDim.x)
			{
				card_next_revised = card_next+blockDim.x 
									- (card_next%blockDim.x); 
			}else{
				card_next_revised = card_next;
			}
		}
		
		index_t lane		= threadIdx.x + strt_pos_curr;
		card_curr			+= strt_pos_curr;
		card_curr_revised	+= strt_pos_curr;

		//cardinality of all vertices in cta_queue
		//is larger than num_threads_in_block
		aq_ver_curr		= adj_list_d[lane];

		cache_ptr	= aq_ver_curr & (HUB_BU_SZ - 1);
		__syncthreads();
		if(__syncthreads_or(aq_ver_curr== hub_cache[cache_ptr]))
		{
			if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;//保证只写一次，让第一个线程去写

			__syncthreads();
			
			ex_ver_curr	= ex_ver_next;
			card_curr 	= card_next;
			card_curr_revised 	= card_next_revised;
			strt_pos_curr= strt_pos_next;
			continue;
		}else{
			adj_depth_curr	= depth_d[aq_ver_curr];
		}
		
		while(lane < card_curr_revised)
		{//后续操作类似
			//-reimburse lane for prefetch
			//-check lane and prefetch for next
			//	iteration
			lane	+= blockDim.x;
			if(lane < card_curr){//0~257 vertices,second round 256, 257
				aq_ver_next	= adj_list_d[lane];
				cache_ptr = aq_ver_next & (HUB_BU_SZ - 1);
			}else{//0~257 vertices, second round 258, ..., 511 vertices
				aq_ver_next	= V_NON_INC;
				cache_ptr = 0;
			}
			
			__syncthreads();
			if(__syncthreads_or(aq_ver_next == hub_cache[cache_ptr]))
			{
				if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;

				break;
			}else{
				if(lane < card_curr) adj_depth_next 
										= depth_d[aq_ver_next];
				else	adj_depth_next	= 0;
			}

			//0	unvisited 	0x00
			//1	fontier		0x01
			//2 visited		0x02
			__syncthreads();
			if(__syncthreads_or(adj_depth_curr == LST_LEVEL))
			{
				if(!threadIdx.x) depth_d[ex_ver_curr] = LEVEL;
				break;
			}
			aq_ver_curr	= aq_ver_next;
			adj_depth_curr= adj_depth_next;
		}
		__syncthreads();
		
		ex_ver_curr	= ex_ver_next;
		card_curr 	= card_next;
		card_curr_revised 	= card_next_revised;
		strt_pos_curr= strt_pos_next;
	}
}


//+------------------------------------------------------
//|Following presents the two types of expanders下面介绍两种类型的扩展器
//|	1. scan_expander
//| 2. sort_expander
//| scan and sort are only used for clarify how they store
//|their expanded results into depth_d扫描和排序仅用于阐明它们如何将扩展结果存储到depth_d中
//+------------------------------------------------------
//|Both of them exploit warp_expander and CTA_expander
//|for expanding different type of ex_queue_d candidates他们都利用warp_expander和CTA_expander来扩展不同类型的ex_queue_d候选者
//+------------------------------------------------------
/*
 Expand ex_queue_d, put all expanded data into depth_d
 by scan offset and adds up展开ex_queue_d，将所有展开的数据按扫描偏移量放入depth_d并加起来
 */

//+----------------------
//|CLFY_EXPAND_SORT
//+----------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
void clfy_expand_sort
(	
	depth_t 		*depth_d,
	index_t			curr_level,
	const vertex_t 	*adj_list_d,
	cudaStream_t 	*stream
)//自上向下的扩展
{
	THD_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td THD ="<<cudaDeviceSynchronize()<<"\n";	
	
	CTA_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td CTA ="<<cudaDeviceSynchronize()<<"\n";	
	
	WAP_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
//	std::cout<<"td WAP ="<<cudaDeviceSynchronize()<<"\n";	
	
}

//+----------------------
//|CLFY_EXPAND_SORT
//+----------------------
template<typename vertex_t, 
		typename index_t,
		typename depth_t>
void clfy_bu_expand_sort
(	
	depth_t 		*depth_d,//每个点的深度
	index_t			curr_level,
	const vertex_t 	*adj_list_d,
	cudaStream_t 	*stream
)
{//三种队列大小，三种并行方式
	//生成自下向上的新的边界点
	THD_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[0]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu THD ="<<cudaDeviceSynchronize()<<"\n";	
		
	WAP_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[1]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu WAP ="<<cudaDeviceSynchronize()<<"\n";	
	
	CTA_bu_expand_sort<vertex_t, index_t, depth_t>
	<<<BLKS_NUM, THDS_NUM, 0, stream[2]>>>
	(
		depth_d,
		curr_level,
		adj_list_d
	);
	//std::cout<<"bu CTA ="<<cudaDeviceSynchronize()<<"\n";	
}

