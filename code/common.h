#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "cudpp.h"
#include "attribute.h"
#include <string>

#define BSIZE (1024)
#define LOG_BSIZE (10)
#ifndef BC
#define PSIZE (6144)
#else
#define PSIZE (4096)
#endif

#define PPSIZE (8192)
#define REG_SIZE (128)
#define RR (REG_SIZE*PSIZE)


#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define CEIL(a,b) (((a)+(b)-1)/(b))
//#define SM_SIZE0 (4096-300+6144) // upper shared memory size, actual_size = 4094
//#define SM_SIZE0 (6144-300+6144) // upper shared memory size, actual_size = 4094
#define SM_SIZE0 (6144)
#define SM_SIZE00 (6144)
#define BG (256*4+150) // buffer granularity
#define SM_SIZE (SM_SIZE0 - BG) // upper shared memory size, actual_size = 4094
//#define BG (256*4+100) // buffer granularity



#define THRESHOLD (6144*4)
#define UIN (100)
#define SUNUSED (UIN + (UIN<<8))
#define IUNUSED (UIN + (UIN<<8) + (UIN<<16) + (UIN<<24))
#define SFACTOR (1)
#define GROUP_NUM (6)
#define GROUP_T (2048)
#define np_base (512)
#define SCATTER_FACTOR (4)
#define DENSE_THRESHOLD (7) // log factor

#define DENSE (0)
#define SPARSE (1)
#define NOT_DETERMINED (2)

#define dsample_factor (2) // *4
#define ssample_factor (8) // *4 
#define ITER_LIMIT (30000)

#define BFS_INF (IUNUSED)
#define SSSP_INF (IUNUSED)

#define PR_INITIAL_VALUE (0.0f)
#define PR_DAMPING_FACTOR (0.85f)
#define PR_TOLERANCE (0.01f)
#define THR (0.05f)

#define INT_PRINT(x,y)  tt = (int *)malloc(sizeof(int)*(y)); \
                        cudaMemcpy(tt, (x), sizeof(int)*(y), cudaMemcpyDeviceToHost);\
                        for(int i=0;i<(y);i++) {\
                          fprintf(stdout, "%d ", tt[i]);\
                        }\
                        fprintf(stdout, "\n");\
                        free(tt);

#define SHORT_PRINT(x,y) tt2 = (short *)malloc(sizeof(short)*(y)); \
                        cudaMemcpy(tt2, (x), sizeof(short)*(y), cudaMemcpyDeviceToHost);\
                        for(int i=0;i<(y);i++) {\
                          fprintf(stdout, "%d ", tt2[i]);\
                        }\
                        fprintf(stdout, "\n");\
                        free(tt2);

#define TEMP_PRINT(x,y) tt2 = (short *)malloc(sizeof(short)*(y)); \
                        cudaMemcpy(tt2, (x), sizeof(short)*(y), cudaMemcpyDeviceToHost);\
                        for(int i=0;i<(y);i++) {\
                          if((i % 100) == 0) fprintf(stdout, "\n");\
                          fprintf(stdout, "%d ", tt2[i]);\
                        }\
                        fprintf(stdout, "\n");\
                        free(tt2);

#define VERTEX_FRONTIER cudaMalloc((void **) &(vd->_curr_f), sizeof(int)*(m->nv)*SCATTER_FACTOR);\
		        cudaMalloc((void **) &(vd->_next_f), sizeof(int)*(m->nv)*SCATTER_FACTOR);\
		        cudaMalloc((void **) &(vd->_gl), sizeof(int)*SCATTER_FACTOR);\
			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);\
			vd->h_fs[0]=0; vd->h_fs[1]=0; vd->h_fs[2]=0; vd->h_fs[3]=0;\
		        cudaMalloc((void **) &(vd->_vv), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);\
		        cudaMalloc((void **) &(vd->_nvv), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);\
		        cudaMemset(vd->_vv, UIN, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);\
		        cudaMemset(vd->_nvv, UIN, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);\
		        cudaMalloc((void **) &(vd->_vfc), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);\
		        cudaMemset(vd->_vfc, -1, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
			//refreshV(m);



#define EDGE_FRONTIER	cudaMalloc((void **) &(vd->_curr_f), sizeof(int)*(m->ne)*2.6);\
		        cudaMalloc((void **) &(vd->_next_f), sizeof(int)*(m->ne)*2.6);\
		        cudaMalloc((void **) &(vd->_gl), sizeof(int)*SCATTER_FACTOR);\
			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);\
			vd->h_fs[0]=0; vd->h_fs[1]=0; vd->h_fs[2]=0; vd->h_fs[3]=0;\
		        cudaMalloc((void **) &(vd->_vv), sizeof(int)*(CEIL(m->ne,PSIZE))*PSIZE);\
		        cudaMemset(vd->_vv, UIN, sizeof(int)*(CEIL(m->ne,PSIZE))*PSIZE);
//		        cudaMalloc((void **) &(vd->_nvv), sizeof(int)*(CEIL(m->ne,PSIZE))*PSIZE);\
//		        cudaMemset(vd->_nvv, UIN, sizeof(int)*(CEIL(m->ne,PSIZE))*PSIZE);\
//		        cudaMalloc((void **) &(vd->_vfc), sizeof(short)*(CEIL(m->ne,PSIZE))*PSIZE);\
//		        cudaMemset(vd->_vfc, -1, sizeof(short)*(CEIL(m->ne,PSIZE))*PSIZE);

		


struct v_struct {
        int src, dst;
        E1T val; // dep
};

struct MULTI_SPARSE {
	int nv, ne;
	int *_p1, *_p2;
	int *_finished;
};

struct HYB {
	int nv, ne, np;
	int upper_nv, upper_np, upper_ne;
	int start_point;

	int *_ncsc_v, *_ncsc_e;
	int *_itable;
	int *_ces, *_cey;
	short *_sp1, *_sp2;
	int *_mapper; 
	int dcnt;
	int *_dx, *_dy, *_dccindex, *_dmap;
	short *_dp1;

#ifdef E1
	E1T *_ncsc_ev;
	E1T *_sp1v;
	E1T *_dp1v;
#endif

	V1T *_i1; //dep
#ifdef V2
	V2T *_i2;
#endif
#ifdef V3
	V3T *_i3;
#endif
	double d_time;

	int dsample_size, dsample_dcnt;
	cudaStream_t stream1, stream2;
	int mp;
	float *_sample_partial;
	int *_temp_front;
	int *_finished;

	int fflag;
	int *degree;
	int ssize;
};

struct csc_package {
	int edges;
        int nv, ne, np;
        int upper_nv, upper_np;
	int DFACTOR, LOG_DFACTOR;
        int *csc_v, *csc_e;
	int *degree;
#ifdef E1
	E1T *csc_ev;
#endif
	int start_p;
	int tra;
};

struct vector_data {
	V1T *_vv, *_nvv, *_dummy_vv;
	int *_vfc;
	int *_curr_f, *_next_f, *_gl;
#ifdef V2
	V2T *_vv2, *_nvv2;
#endif
#ifdef V3
	V3T *_vv3;
#endif
//	int *_temp_f;
	int htot_size, h_fs[SCATTER_FACTOR];
	int algo, fhave;
	int *accum_frpoint, *accum_fr;
	int last_accum;
};

//struct MG {
//	int init;
//};

struct OUTPUT {
	V1T *r;
#ifdef V2
	V2T *r2;
#endif
#ifdef V3
	V3T *r3;
#endif
};

void generate_CSC(struct csc_package *, char **);

double rtclock();
__device__ int atomicAggInc(int *);
__device__ short atomicAddShort(short *,short);

void generate_MultiGraph(struct HYB *, struct csc_package *);
void generate_MultiGraphS(struct MULTI_SPARSE *, struct csc_package *);

void release_MultiGraph(struct HYB *);

//void initValue(struct HYB *, int *);

int MultiGraph_V_E_V(struct HYB *, struct csc_package *, struct vector_data *, struct OUTPUT *, int *);
int MultiGraph_V_E_V_pull(struct HYB *, struct csc_package *, struct vector_data *, struct OUTPUT *, int *);
int MultiGraph_V_E_V_NOTIDEM(struct HYB *, struct csc_package *, struct vector_data *, struct OUTPUT *, int *);

int MultiGraph_E_E(struct MULTI_SPARSE *, struct vector_data *, struct OUTPUT *, int *);
int MultiGraph_V_V(struct MULTI_SPARSE *, struct vector_data *, struct vector_data *, struct OUTPUT *, int *);

void verifyResults(struct csc_package *, GRAPHTYPE *, struct OUTPUT *, int);

//void resetValue(int , GRAPHTYPE *m, struct vector_data *);

__device__ inline int lane_id(void) { return (threadIdx.x&31); }

__device__ inline int warp_bcast(int v, int leader) { return __shfl(v, leader); }

__device__ inline int atomicAggInc(int *ctr) {
        int mask = __ballot(1);
        int leader = __ffs(mask) - 1;
        int res;
        if(lane_id() == leader)
                res = atomicAdd(ctr, __popc(mask));
        res = warp_bcast(res, leader);

        return (res + __popc(mask & ((1 << lane_id()) - 1)));
}

__device__ inline int atomicPSeg(int *ctr, int pnt) {
        int npnt = __shfl_up(pnt, 1);
        int mask, nextlane, plane=-1;
        int v;
	int mask0 = __ballot(1);
	int leader = __ffs(mask0) - 1; 
        if(lane_id() == leader || pnt != npnt || (mask0 & (1<< (lane_id()-1))) == 0) {
                mask = __ballot(1);
                nextlane = __ffs(mask & (0xffffffff - ((2 << lane_id()) - 1))) - 1;
                if(nextlane == -1) nextlane = 32;
                plane = lane_id();
		int size = __popc(mask0 & ((1 << nextlane) - (1 << lane_id())));  
                v = atomicAdd(ctr, size);
        }
        mask = __shfl(mask, leader);
        if(plane < 0) plane = 32 - __ffs(__brev(mask) & (0xffffffff - (0xffffffff >> lane_id())));
        return( __shfl(v, plane) + (lane_id() - plane));
}


__device__ inline int atomicSeg(int *ctr, int pnt) {
        int npnt = __shfl_up(pnt, 1);
        int mask, nextlane, plane=-1;
        int v;
	mask = __ballot(1);
        if(lane_id() == 0 || pnt != npnt) {
                mask = __ballot(1);
                nextlane = __ffs(mask & (0xffffffff - ((2 << lane_id()) - 1))) - 1;
                if(nextlane == -1) nextlane = 32;
                plane = lane_id();
                v = atomicAdd(ctr, nextlane - lane_id());
        }
        mask = __shfl(mask, 0);
        if(plane < 0) plane = 32 - __ffs(__brev(mask) & (0xffffffff - (0xffffffff >> lane_id())));
        return( __shfl(v, plane) + (lane_id() - plane));
}





__device__ inline short atomicAddShort(short* address, short val) {
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
}

__device__ inline short atomicMaxShort(short* address, short val) {
        unsigned int *base_address = (unsigned int *) ((char *)address - ((size_t)address & 2));
        unsigned int long_val = ((size_t)address & 2) ? ((unsigned int)val << 16) : (unsigned short)val;
        unsigned int long_old = atomicMax(base_address, long_val);
        if((size_t)address & 2) {
                return (short)(long_old >> 16);
        } else {
                unsigned int overflow = ((long_old & 0xffff) + long_val) & 0xffff0000;
                if (overflow)
                        atomicSub(base_address, overflow);
                return (short)(long_old & 0xffff);
        }
}



__device__ inline double atomicAddDouble(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


////////////////////////////////////
/*
__device__ inline void initialize(int *loc, int data)
{
        *loc = BFS_INF;
}

__device__ inline void get_partial_result(int *loc, int data)
{
        *loc = data+1;
}

__device__ inline void accumulate(int *loc, int data)
{
        *loc = MIN(*loc, data);
}

__device__ inline int update_condition(int *loc, int data)
{
        if(data < *loc) {
                *loc = data;
                return true;
        } else {
                return false;
	}
}*/

