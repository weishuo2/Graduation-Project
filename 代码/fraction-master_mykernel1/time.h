#ifndef _TIME_H_
#define _TIME_H_

#include <sys/time.h>
//计时
#define elapsed(tv1, tv2) \
    ((tv2.tv_sec - tv1.tv_sec) * 1000 + (double)(tv2.tv_usec - tv1.tv_usec) / 1000)

#endif
