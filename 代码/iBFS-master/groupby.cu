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

#include "util.h"
#include "graph.h"
#include <iostream>
#include <fstream>


void
graph::
groupby()
{//将输入的源点进行对应的分组，使得每组的源点有尽可能多的共同边界点
	vertex_t *new_list = new index_t[src_count];
	index_t *hubmap=new index_t[vert_count];
	index_t *huboff=new index_t[vert_count];
	memset(hubmap,0,sizeof(index_t)*vert_count);
	memset(huboff,0,sizeof(index_t)*vert_count);
	
	//-divide source into connected to hub源点应该连接到中心顶点
	//-not connected to half by "remainoff"没连到的标记为剩余
	/*counting eligible instances*/
	//计算合格的实例
	for(index_t i=0;i<src_count;i++)
	{
		index_t beg=beg_pos[src_list[i]];
		index_t end=beg_pos[src_list[i]+1];
		
		/*out-degree <= sdegree*/
		if(end-beg>128) continue; //出度过大
		for(index_t j=beg;j<end;j++)
		{
			/*has hub neighbor*/
			//邻居为中心点
			if(beg_pos[csr[j]+1]-
				beg_pos[csr[j]] > 128)
			{
				hubmap[csr[j]]++;//记录有多少个源点链接到这个点
				break;
			}
		}
	}
	
	/*prefix-sum counted data*/
	//前缀和计数数据
	huboff[0]=0;//前缀和
	for(index_t i=1;i<vert_count;i++)
		huboff[i]=huboff[i-1]+hubmap[i-1];
	index_t remainoff=huboff[vert_count-1]
			+hubmap[vert_count-1];//连接到中心顶点且自身出度较小的源点数目
	
	index_t test=0;
	/*formulating group*/
	//制定组  
	for(index_t i=0;i<src_count;i++)
	{
		index_t beg=beg_pos[src_list[i]];
		index_t end=beg_pos[src_list[i]+1];
		
		//no hub neighbor, put it after "remainoff"没有中心点
		/*non-eligible instance*/
		//不合适的实例
		if(end-beg>128)
		{
			new_list[remainoff]=src_list[i];
			remainoff++;
			test++;
			continue;
		}
		
		/*eligible instance*/
		//合适的实例
		bool isEligible=false;
		for(index_t j=beg;j<end;j++)
		{
			//-put it in the corresponding position of 
			//-prefix sum
			/*has hub neighbor*/
			//中心点邻居
			if(beg_pos[csr[j]+1]-
				beg_pos[csr[j]] > 128)
			{//很巧妙的运用前缀和，将连接到同一个中心点的源点放在一起
				new_list[huboff[csr[j]]]=src_list[i];
				huboff[csr[j]]++;
				isEligible=true;
				break;
			}
		}
		
		if(!isEligible)
		{//说明虽然出度少，但是没连到中心点，还是不符合
			new_list[remainoff]=src_list[i];
			remainoff++;
			test++;
		}
	}

	#ifdef ENABLE_MONITORING
	std::ofstream myfile("origin_src.dat");
	std::ofstream newfile("new_src.dat");
	for(index_t i=0;i<src_count;i++)
	{//写入文件
		myfile<<src_list[i]<<"\n";
		newfile<<new_list[i]<<"\n";
	}
	myfile.close();
	newfile.close();

	index_t remain=src_count;
	index_t grouped=0;
	std::cout<<"Eligible-BFS(by-src-remain): ";
	for(index_t i=0;i<vert_count;i++)
	{
		if(hubmap[i])
		{
			std::cout<<hubmap[i]<<" ";
			grouped+=hubmap[i];
		}
	}
	remain-=grouped;
	std::cout<<remain<<"\n";
	std::cout<<"Get test: "<<test<<"\n";
	#endif
	
	delete[] src_list;
	delete[] hubmap;
	delete[] huboff;
	src_list=new_list;
	return;
}
