#include <stack>
#include <string.h>
#include <stdlib.h>
#include <cassert>
#include <iostream>

inline void bfs(const int * row_ptr, const int * col, const int * col_ptr, const int * row, const int * is_target, const int src, int * visited)
{
    if (is_target[src] == 0)
        return;

    std::stack<int> s;
    s.push(src);
    visited[src] = 1;

    while(!s.empty())
    {
        int cur_node = s.top();
        s.pop();
        for (int dst_ind = row_ptr[cur_node]; dst_ind < row_ptr[cur_node+1]; dst_ind++)
        {
            int dst = col[dst_ind];
            if (is_target[dst] == 0 || visited[dst])
                continue;
            s.push(dst);
            visited[dst] = 1;
        }

        for (int src_ind = col_ptr[cur_node]; src_ind < col_ptr[cur_node+1]; src_ind++)
        {
            int src_v = row[src_ind];
            if (is_target[src_v] == 0 || visited[src_v])
                continue;
            s.push(src_v);
            visited[src_v] = 1;
        }
    }
}

int getNumCC(const int n, const int * row_ptr, const int * col, const int * col_ptr, const int * row, const int * is_target)
{//判断有多少个子图
    int num_CC = 0;
    int * visited = (int *)malloc(sizeof(int) * n);
    memset(visited, 0, sizeof(int) * n);
    for (int i = 0; i < n; ++i)
    {
        if (is_target[i] == 0 || visited[i] == 1)
            continue;
        bfs(row_ptr, col, col_ptr, row, is_target, i, visited);
        num_CC++;
    }
    for (int i = 0; i < n; ++i)
    {
        assert(visited[i] == is_target[i]);
    }
    free(visited);
    return num_CC;
}
