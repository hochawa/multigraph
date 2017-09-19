//#include "common.h"
#include "user_function.h"

double rtclock(void)
{
  struct timezone Tzp;
  struct timeval Tp;
  int stat;
  stat = gettimeofday (&Tp, &Tzp);
  if (stat != 0) printf("Error return from gettimeofday: %d",stat);
  return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

/*
__device__ inline int lane_id(void) { return (threadIdx.x&31); }

__device__ int warp_bcast(int v, int leader) { return __shfl(v, leader); }

__device__ int atomicAggInc(int *ctr) {
        int mask = __ballot(1);
        int leader = __ffs(mask) - 1;
        int res;
        if(lane_id() == leader)
                res = atomicAdd(ctr, __popc(mask));
        res = warp_bcast(res, leader);

        return (res + __popc(mask & ((1 << lane_id()) - 1)));
}

__device__ short atomicAddShort(short* address, short val) {
        unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));
        unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
        unsigned int long_old = atomicAdd(base_address, long_val);
        if((size_t)address & 2) {
                return (short)(long_old >> 16);
        } else {
                unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
                if (overflow)
                        atomicSub(base_address, overflow);
                return (short)(long_old & 0xffff);
        }
}*/
