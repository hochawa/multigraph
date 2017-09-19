#include "user_function.h"
//#include "common.h"

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

__global__ void s_preprocessing(int nv, int *csc_v, int *csc_e, int *p1, int *p2)
{
	__shared__ int buffer1[64], buffer2[64], buffer3[64], buffer4[64];
	__shared__ int buffer_p[2];

	int i, j;
	int index = blockIdx.x*64 + (threadIdx.x>>2), bias;

	if(threadIdx.x < 2) {
		buffer_p[threadIdx.x] = 0;
	}
	__syncthreads();

	if(index < nv) {
		bias = 0;		
		int index_size = csc_v[index+1] - csc_v[index];

	        if(index_size >= 32) {
	                bias = index_size - (index_size&31);
	                if((threadIdx.x&3) == 0) {
        	                int p = atomicAggInc(&buffer_p[0]);
	                        buffer1[p] = index;
	                        buffer2[p] = bias;
	                }
	                if(index_size >= 256) {
	                        if((threadIdx.x&3) == 0) {
        	                        int p2 = atomicAggInc(&buffer_p[1]);
	                                buffer3[p2] = index;
	                                buffer4[p2] = index_size - (index_size&255);
	                        }
	                }
        	}

		for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
			int dst0 = csc_v[index]+i;
			int index_dst = csc_e[dst0];
			p1[dst0] = index;
			p2[dst0] = index_dst;
		}
	}
	__syncthreads();

	for(i=(threadIdx.x>>5);i<buffer_p[0];i+=8) {
		index = buffer1[i];
		int bf2 = buffer2[i];
		int bf22 = bf2 - (bf2&255);
		for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
			int dst0 = csc_v[index]+j;
			int index_dst = csc_e[dst0];
			p1[dst0] = index;
			p2[dst0] = index_dst;
		}
	}
	for(i=0;i<buffer_p[1];i++) {
		index = buffer3[i];
		for(j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
			int dst0 = csc_v[index]+j;
			int index_dst = csc_e[dst0];
			p1[dst0] = index;
			p2[dst0] = index_dst;
		
		}
	}
}

void generate_MultiGraphS(struct MULTI_SPARSE *m, struct csc_package *inp)
{

int *tt;
	cudaError_t cuda_stat;

	(m->nv) = inp->nv, (m->ne) = inp->ne;
	int *csc_v = inp->csc_v, *csc_e = inp->csc_e;
	int *_csc_v, *_csc_e;
	int *_p1, *_p2;

	cuda_stat = cudaMalloc((void **) &_csc_v, sizeof(int)*((inp->nv)+1));
	cuda_stat = cudaMalloc((void **) &_csc_e, sizeof(int)*(inp->ne));
	cudaMemcpy(_csc_v, csc_v, sizeof(int)*((inp->nv)+1), cudaMemcpyHostToDevice);
	cudaMemcpy(_csc_e, csc_e, sizeof(int)*(inp->ne), cudaMemcpyHostToDevice);
	cuda_stat = cudaMalloc((void **) &_p1, sizeof(int)*(inp->ne));
	cuda_stat = cudaMalloc((void **) &_p2, sizeof(int)*(inp->ne));

	cudaDeviceSynchronize();
//for(int i=0;i<(inp->ne);i++) {
//	printf("%d ", inp->csc_e[i]);
//} printf("\n");

//INT_PRINT(_csc_v, (m->nv)+1);
//INT_PRINT(_csc_e, m->ne);

	double s_time=rtclock();
	cudaDeviceSynchronize();
	s_preprocessing<<<((m->nv)+255)>>6, 256>>>(m->nv, _csc_v, _csc_e, _p1, _p2);
	cudaDeviceSynchronize();
	double e_time=rtclock();
	fprintf(stdout, "preprocessing : %f ms,", (e_time - s_time)*1000);

	m->_p1 = _p1; m->_p2 = _p2;
	cuda_stat = cudaMalloc((void **) &(m->_finished), sizeof(int));

//INT_PRINT(_csc_v, (m->nv)+1);
//INT_PRINT(_csc_e, m->ne);
//INT_PRINT(_p1, m->ne);
//INT_PRINT(_p2, m->ne);
}


__global__ void preprocessing_step000(int nv, int *csc_v, short *csc_occ)
{
        for(int index =(blockDim.x*blockIdx.x*SFACTOR)+(threadIdx.x&31)+((threadIdx.x>>5)<<5)*SFACTOR; index < nv; index += blockDim.x*gridDim.x*SFACTOR) {
//              if(csc_occ[csc_e[index]] < GROUP_T) atomicAddShort(&csc_occ[csc_e[index]], SFACTOR);
                int t = MIN(GROUP_T, csc_v[index+1] - csc_v[index]);
		csc_occ[index] = t;
        }
}


__global__ void preprocessing_step00(int ne, int *csc_e, short *csc_occ)
{
        for(int index =(blockDim.x*blockIdx.x*SFACTOR)+(threadIdx.x&31)+((threadIdx.x>>5)<<5)*SFACTOR; index < ne; index += blockDim.x*gridDim.x*SFACTOR) {
                if(csc_occ[csc_e[index]] < GROUP_T) atomicAddShort(&csc_occ[csc_e[index]], SFACTOR);
        }
}


__global__ void preprocessing_step01(int nv, int upper_nv, short *csc_occ, int *group_occ, int *gr) {
	__shared__ int sgroup_occ[6144];

	int i, j, bias;
	for(i=threadIdx.x;i<6144;i+=blockDim.x) {
		sgroup_occ[i] = 0;
	} 
	__syncthreads();

	for(i=blockDim.x*blockIdx.x+threadIdx.x;i<nv;i+=gridDim.x*blockDim.x) {
		short k = csc_occ[i];
		//if(k > 4096) printf("err : %d\n", k);
		if(k >= GROUP_T) {
			bias=0;
			atomicAdd(&sgroup_occ[GROUP_NUM*threadIdx.x], 1);
		}
		else if(k >= (GROUP_T>>1)) {
			bias=upper_nv;
			atomicAdd(&sgroup_occ[GROUP_NUM*threadIdx.x+1], 1);
		}
		else if(k >= (GROUP_T>>2)) {
			bias=upper_nv*2;
			atomicAdd(&sgroup_occ[GROUP_NUM*threadIdx.x+2], 1);
		}
		else if(k >= (GROUP_T>>3)) {
			bias=upper_nv*3;
			atomicAdd(&sgroup_occ[GROUP_NUM*threadIdx.x+3], 1);
		}
		else if(k >= (GROUP_T>>4)) {
			bias=upper_nv*4;
			atomicAdd(&sgroup_occ[GROUP_NUM*threadIdx.x+4], 1);
		} else {
			bias=upper_nv*5;
		}
		gr[i+bias] = 1;
	}
	__syncthreads();

	//reduction
	for(j=3072;j>=6;j>>=1) {
		for(i=threadIdx.x;i<j;i+=blockDim.x) {
			sgroup_occ[i] += sgroup_occ[i+j];
		}
		 __syncthreads();
	}
	if(threadIdx.x < GROUP_NUM) {
		atomicAdd(&group_occ[threadIdx.x], sgroup_occ[threadIdx.x]);
	}
}

__global__ void preprocessing_step02(int nv, int upper_nv, const int * __restrict__ csc_v, int *cgr, int *itable, int *csc_size, short *csc_occ,
int loc1, int loc2, int loc3, int loc4, int loc5)
{
//	int index=blockDim.x*blockIdx.x+threadIdx.x;
	for(int index=blockDim.x*blockIdx.x+threadIdx.x;index<nv;index+=blockDim.x*gridDim.x) {
		short k = csc_occ[index];
		int t;
		if(k >= GROUP_T) {
			itable[index] = cgr[index];
		} else if(k >= (GROUP_T>>1)) {
			itable[index] = loc1+cgr[index+upper_nv];
		} else if(k >= (GROUP_T>>2)) {
			itable[index] = loc2+cgr[index+upper_nv*2];
		} else if(k >= (GROUP_T>>3)) {
			itable[index] = loc3+cgr[index+upper_nv*3];
		} else if(k >= (GROUP_T>>4)) {
			itable[index] = loc4+cgr[index+upper_nv*4];
		} else {
			itable[index] = loc5+cgr[index+upper_nv*5];
		}
//if(itable[index] != index) printf("err : %d\n", itable[index], index);

		t = itable[index];
/*
if(t < 0 || t >= nv) {
int kkkk;
if(k >= GROUP_T) {
kkkk = cgr[index]-1;
} else if(k >= (GROUP_T>>1)) {
kkkk = loc1+cgr[index+upper_nv]-1;
} else if(k >= (GROUP_T>>2)) {
kkkk = loc2+cgr[index+upper_nv*2]-1;
} else if(k >= (GROUP_T>>3)) {
kkkk = loc3+cgr[index+upper_nv*3]-1;
} else if(k >= (GROUP_T>>4)) {
kkkk = loc4+cgr[index+upper_nv*4]-1;
} else {
kkkk = loc5+cgr[index+upper_nv*5]-1;
}

printf("err %d %d %d %d\n", index, csc_occ[index], kkkk, t);
}*/
		csc_size[t] = csc_v[index+1] - csc_v[index];
	}
}




__global__ void preprocessing_step03(int nv, int niter, const int * __restrict__ itable, int *csc_v, int *csc_e, int *ncsc_v, int *tmp_e, int *ncsc_e,
int g1, int g2, int g3, int g4, int g5
#ifdef E1
, E1T *csc_ev, E1T *ncsc_ev
#endif
)
{
        __shared__ int buffer1[64], buffer2[64], buffer22[64], buffer3[64];
        __shared__ int buffer01[64];
        __shared__ int buffer_p[2];
        __shared__ int sleft_v[64];

//      __shared__ int sr0[64], sr1[64], sr2[64], sr3[64], sr4[64], sr5[64];
        __shared__ int sr[64*6];

        int index, index_size, diff;
        int i, j, bias=0;
        int r0, r1, r2, r3, r4, r5;

        index=blockIdx.x*64+(threadIdx.x>>1);
//      for(index=blockIdx.x*64+(threadIdx.x>>2); index<niter; index+=gridDim.x*64) {
        for(i=threadIdx.x;i<64*6;i+=blockDim.x) {
                sr[i] = 0;
        }

                bias = 0;
                if(threadIdx.x < 2) {
                        buffer_p[threadIdx.x] = 0;
                }
                __syncthreads();
                if(index < nv) {
                        int right_v = csc_v[index];
                        index_size = csc_v[index+1] - right_v;
                        int left_v = ncsc_v[itable[index]];
                        if((threadIdx.x&1) == 0) {
                                sleft_v[(threadIdx.x>>1)] = left_v;
                        }
//                      diff = ncsc_v[itable[index]] - csc_v[index];

                        bias = (index_size&31);
                        if(index_size >= 32) {
//                              bias = index_size - (index_size&31);
                                if((threadIdx.x&1) == 0) {
                                        int p = atomicAggInc(&buffer_p[0]);
                                        buffer1[p] = (threadIdx.x>>1);
                                        buffer2[p] = bias;
                                        buffer22[p] = index_size;
                                        buffer3[p] = left_v;
                                        buffer01[p] = right_v;
                                }
                        }
                        r0 = r1 = r2 = r3 = r4 = r5 = 0;
                        for(i=(threadIdx.x&1); i<bias; i+=2) {
                                int tmp = itable[csc_e[right_v + i]];
                                if(tmp < g1) r0++;
                                else if(tmp < g2) r1++;
                                else if(tmp < g3) r2++;
                                else if(tmp < g4) r3++;
                                else if(tmp < g5) r4++;
//                              else r5++;

                                tmp_e[left_v + i] = tmp;
#ifdef E1
//                              ncsc_ev[left_v + i] = csc_ev[right_v + i];
#endif
                        }

                        r0 += __shfl_down(r0, 1);
                        r1 += __shfl_down(r1, 1);
                        r2 += __shfl_down(r2, 1);
                        r3 += __shfl_down(r3, 1);
                        r4 += __shfl_down(r4, 1);
//                      r5 += __shfl_down(r5, 1, 4);

                        if((threadIdx.x&1) == 0) {
                                sr[6*(threadIdx.x>>1)] = r0;
                                sr[6*(threadIdx.x>>1)+1] = r1;
                                sr[6*(threadIdx.x>>1)+2] = r2;
                                sr[6*(threadIdx.x>>1)+3] = r3;
                                sr[6*(threadIdx.x>>1)+4] = r4;
//                              sr[320+(threadIdx.x>>2)] = r5;

                        }

                }
                __syncthreads();

                for(i=(threadIdx.x>>5);i<buffer_p[0];i+=4) {
//                      int lindex = buffer1[i];
                        int bf2 = buffer2[i];
                        int bf22 = buffer22[i];
                        int bf3 = buffer3[i];
                        int bf4 = buffer01[i];
                        r0 = r1 = r2 = r3 = r4 = r5 = 0;
                        for(j=bf2+(threadIdx.x&31);j<bf22;j+=32) {
                                int tmp = itable[csc_e[bf4+j]];
                                if(tmp < g1) r0++;
                                else if(tmp < g2) r1++;
                                else if(tmp < g3) r2++;
                                else if(tmp < g4) r3++;
                                else if(tmp < g5) r4++;
//                              else r5++;

                                tmp_e[j+bf3] = tmp;
#ifdef E1
//                              ncsc_ev[j+bf3] = csc_ev[bf4+j];
#endif
                        }
                        r0 += __shfl_down(r0, 16);
                        r1 += __shfl_down(r1, 16);
                        r2 += __shfl_down(r2, 16);
                        r3 += __shfl_down(r3, 16);
                        r4 += __shfl_down(r4, 16);
//                      r5 += __shfl_down(r5, 16, 32);

                        r0 += __shfl_down(r0, 8);
                        r1 += __shfl_down(r1, 8);
                        r2 += __shfl_down(r2, 8);
                        r3 += __shfl_down(r3, 8);
                        r4 += __shfl_down(r4, 8);
//                      r5 += __shfl_down(r5, 8);

                        r0 += __shfl_down(r0, 4);
                        r1 += __shfl_down(r1, 4);
                        r2 += __shfl_down(r2, 4);
                        r3 += __shfl_down(r3, 4);
                        r4 += __shfl_down(r4, 4);
//                      r5 += __shfl_down(r5, 4);

                        r0 += __shfl_down(r0, 2);
                        r1 += __shfl_down(r1, 2);
                        r2 += __shfl_down(r2, 2);
                        r3 += __shfl_down(r3, 2);
                        r4 += __shfl_down(r4, 2);

                        r0 += __shfl_down(r0, 1);
                        r1 += __shfl_down(r1, 1);
                        r2 += __shfl_down(r2, 1);
                        r3 += __shfl_down(r3, 1);
                        r4 += __shfl_down(r4, 1);
//                      r5 += __shfl_down(r5, 1);

                        if((threadIdx.x&31) == 0) {
                                sr[6*buffer1[i]] += r0;
                                sr[6*buffer1[i]+1] += r1;
                                sr[6*buffer1[i]+2] += r2;
                                sr[6*buffer1[i]+3] += r3;
                                sr[6*buffer1[i]+4] += r4;
//                              sr[6*buffer1[i]+5] += r5;
                        }

                }

                __syncthreads();


        if(threadIdx.x < 64) {
                sr[6*threadIdx.x + 1] += sr[6*threadIdx.x];
                sr[6*threadIdx.x + 2] += sr[6*threadIdx.x+1];
                sr[6*threadIdx.x + 3] += sr[6*threadIdx.x+2];
                sr[6*threadIdx.x + 4] += sr[6*threadIdx.x+3];

                sr[6*threadIdx.x + 5] = sr[6*threadIdx.x+4];
                sr[6*threadIdx.x + 4] = sr[6*threadIdx.x+3];
                sr[6*threadIdx.x + 3] = sr[6*threadIdx.x+2];
                sr[6*threadIdx.x + 2] = sr[6*threadIdx.x+1];
                sr[6*threadIdx.x + 1] = sr[6*threadIdx.x];
                sr[6*threadIdx.x] = 0;
        }
                __syncthreads();



        //actual moving

        index=blockIdx.x*64+threadIdx.x;
//      for(index=blockIdx.x*64+(threadIdx.x>>2); index<niter; index+=gridDim.x*64) {

                if(index < nv && threadIdx.x < 64) {
                        int right_v = csc_v[index];
                        int left_v = sleft_v[threadIdx.x];
                        index_size = csc_v[index+1] - right_v;

                        bias = (index_size&31);


                        for(i=0; i<bias; i++) {
                                int tmp = tmp_e[left_v + i];
                                int loc;
                                if(tmp < g1) {
                                        loc = sr[6*threadIdx.x];
                                        sr[6*threadIdx.x]++;
                                } else if(tmp < g2) {
                                        loc = sr[6*threadIdx.x+1];
                                        sr[6*threadIdx.x+1]++;
                                } else if(tmp < g3) {
                                        loc = sr[6*threadIdx.x+2];
                                        sr[6*threadIdx.x+2]++;
                                } else if(tmp < g4) {
                                        loc = sr[6*threadIdx.x+3];
                                        sr[6*threadIdx.x+3]++;
                                } else if(tmp < g5) {
                                        loc = sr[6*threadIdx.x+4];
                                        sr[6*threadIdx.x+4]++;
                                } else {
                                        loc = sr[6*threadIdx.x+5];
                                        sr[6*threadIdx.x+5]++;
                                }

                                ncsc_e[left_v + loc] = tmp;
#ifdef E1
                                ncsc_ev[left_v + loc] = csc_ev[right_v + i];
#endif
                        }
                }
                __syncthreads();

                for(i=(threadIdx.x>>5);i<buffer_p[0];i+=4) {
                      int lindex = buffer1[i];
                        int bf2 = buffer2[i];
                        int bf22 = buffer22[i];
                        int bf3 = buffer3[i];
                        int bf4 = buffer01[i];
                        for(j=bf2+(threadIdx.x&31);j<bf22;j+=32) {
                                int tmp = tmp_e[bf3+j];
                                int loc;

//                              if(tmp < g1) loc = atomicAdd(&sr[6*lindex],1);
//                              else if(tmp < g2) loc = atomicAdd(&sr[6*lindex+1],1);
//                              else if(tmp < g3) loc = atomicAdd(&sr[6*lindex+2],1);
//                              else if(tmp < g4) loc = atomicAdd(&sr[6*lindex+3],1);
//                              else if(tmp < g5) loc = atomicAdd(&sr[6*lindex+4],1);
//                              else loc = atomicAdd(&sr[6*lindex+5],1);

                                if(tmp < g1) loc = atomicAggInc(&sr[6*lindex]);
                                if(tmp >= g1 && tmp < g2) loc = atomicAggInc(&sr[6*lindex+1]);
                                if(tmp >= g2 && tmp < g3) loc = atomicAggInc(&sr[6*lindex+2]);
                                if(tmp >= g3 && tmp < g4) loc = atomicAggInc(&sr[6*lindex+3]);
                                if(tmp >= g4 && tmp < g5) loc = atomicAggInc(&sr[6*lindex+4]);
                                if(tmp >= g5) loc = atomicAggInc(&sr[6*lindex+5]);



//                              if(tmp < g1) loc = atomicAggInc(&sr[6*lindex]);
//                              else if(tmp < g2) loc = atomicAggInc(&sr[6*lindex+1]);
//                              else if(tmp < g3) loc = atomicAggInc(&sr[6*lindex+2]);
//                              else if(tmp < g4) loc = atomicAggInc(&sr[6*lindex+3]);
//                              else if(tmp < g5) loc = atomicAggInc(&sr[6*lindex+4]);
//                              else loc = atomicAggInc(&sr[6*lindex+5]);

                                ncsc_e[loc+bf3] = tmp;
#ifdef E1
                                ncsc_ev[loc+bf3] = csc_ev[bf4+j];

#endif
			}
		}
}




















__global__ void preprocessing_step03r(int nv, int niter, const int * __restrict__ itable, const int *csc_v, int *csc_e, const int *ncsc_v, int *ncsc_e
#ifdef E1
, E1T *csc_ev, E1T *ncsc_ev
#endif
)
{
	__shared__ int buffer1[64], buffer2[64], buffer3[64], buffer4[64], buffer5[64], buffer6[64];
	__shared__ int buffer01[64], buffer02[64];
	__shared__ int buffer_p[2];

	int index, index_size, diff;
	int i, j, bias=0;

	index=blockIdx.x*64+(threadIdx.x>>2);
//	for(index=blockIdx.x*64+(threadIdx.x>>2); index<niter; index+=gridDim.x*64) {

		bias = 0;
		if(threadIdx.x < 2) {
			buffer_p[threadIdx.x] = 0;
		}
		__syncthreads();
		if(index < nv) {
			int right_v = csc_v[index];	
			index_size = csc_v[index+1] - right_v;		
			int left_v = right_v + ncsc_v[itable[index]] - right_v;
//			diff = ncsc_v[itable[index]] - csc_v[index];
	
		        if(index_size >= 32) {
				bias = index_size - (index_size&31);
		                if((threadIdx.x&3) == 0) {
		                        int p = atomicAggInc(&buffer_p[0]);
  //     					buffer1[p] = index;
       	                 	buffer2[p] = bias;
					buffer3[p] = left_v; 
					buffer01[p] = right_v;      
 	         	}
       	         	if(index_size >= 256) {
       	                 	if((threadIdx.x&3) == 0) {
       	                      		int p2 = atomicAggInc(&buffer_p[1]);
     //  	                        		buffer4[p2] = index;
       	                         	buffer5[p2] = index_size - (index_size&255);
						buffer6[p2] = left_v;
						buffer02[p2] = right_v;
       	                 	}
       	    	 	}
       	 	}
			for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
				ncsc_e[left_v + i] = itable[csc_e[right_v + i]];
#ifdef E1
				ncsc_ev[left_v + i] = csc_ev[right_v + i];
				
#endif
			}
		}
		__syncthreads();
	
		for(i=(threadIdx.x>>5);i<buffer_p[0];i+=8) {
//			int lindex = buffer1[i];
			int bf2 = buffer2[i];
			int bf22 = bf2 - (bf2&255);
			int bf3 = buffer3[i];
			int bf4 = buffer01[i];
			for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
				ncsc_e[j+bf3] = itable[csc_e[bf4+j]];
#ifdef E1
				ncsc_ev[j+bf3] = csc_ev[bf4+j];

#endif
			} 
		}
		for(i=0;i<buffer_p[1];i++) {
//			int lindex = buffer4[i];
			int bf3 = buffer6[i];
			int bf4 = buffer02[i];
			for(j=threadIdx.x;j<buffer5[i];j+=blockDim.x) {
				ncsc_e[j+bf3] = itable[csc_e[bf4+j]];
#ifdef E1
				ncsc_ev[j+bf3] = csc_ev[bf4+j];

#endif
			}
		}
//		__syncthreads();

//	}
	
}


__global__ void
preprocessing_step1verybig(int *buff, int DFACTOR, int LOG_DFACTOR, int nv, int ne, int np, int upper_np, int *csc_v, const int * __restrict__ csc_e, int *es, int *pb1, int *pb2,
int *dcnt, int *dindex, int *dx, int *dy)
{

	__shared__ int occ[SM_SIZE00];
//	__shared__ int buffer1[BG], buffer3[BG];
//	__shared__ int buffer_p[2];

	int i, j, k, index, index_size, bias;

//	int dd=0, d0=0, d1=0, d2=0, d3=0;
//	int d4=0, d5=0, d6=0, d7=0;
	int dd=0;

	// get occurrance
	for(i=threadIdx.x;i<upper_np;i+=blockDim.x) {
		occ[i] = 0;
	}

	__syncthreads();
	for(index=blockIdx.x*PSIZE+(threadIdx.x>>0); index-blockIdx.x*PSIZE < PSIZE; ) {
//		__syncthreads();
		for(k=0;k<1;k++) {
			if(index < nv) {
				index_size = csc_v[index+1] - csc_v[index];
				// step 1 (small granularity)
				for(i=0; i<index_size; i++) {
//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("etype1 %d %d\n", index, csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("etype2 %d %d\n", index, csc_e[csc_v[index]+i]);
					int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE00) printf("etype3 : %d %d\n", index, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
					int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//atomicPSeg(&occ[index_pnt], index_pnt);
					if(index_pnt != blockIdx.x) {
						atomicPSeg(&occ[ti], ti);
					} else {
						dd++;
					}
	

				//occ[index_pnt]++;
				}
			}
			index += 1024;
		}
		//__syncthreads();
	}

        dd += __shfl_down(dd, 16);
        dd += __shfl_down(dd, 8);
        dd += __shfl_down(dd, 4);
        dd += __shfl_down(dd, 2);
        dd += __shfl_down(dd, 1);

        if((threadIdx.x&31)==0) {
                occ[SM_SIZE00-3-(threadIdx.x>>5)] = dd;
        }
        __syncthreads();
        if(threadIdx.x<32) {
                dd = occ[SM_SIZE00-3-threadIdx.x];
                dd += __shfl_down(dd, 16);
                dd += __shfl_down(dd, 8);
                dd += __shfl_down(dd, 4);
                dd += __shfl_down(dd, 2);
                dd += __shfl_down(dd, 1);
        }
        __syncthreads();
	
	if(threadIdx.x == 0) {
		occ[DFACTOR*blockIdx.x] = dd;
	}

	for(i=DFACTOR*np+threadIdx.x;i<SM_SIZE00;i+=blockDim.x) {
		occ[i] = 0;
	}
	__syncthreads();

	//make dense-information, normalization
	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		int t=0;
		for(j=0;j<DFACTOR-1;j++) {
			t += occ[i*DFACTOR+j];
			occ[i*DFACTOR+j] = 0;
		}
		int t2=t+occ[i*DFACTOR+DFACTOR-1];
		occ[i*DFACTOR+DFACTOR-1] = 0;
		t2 = (((t2+7)>>3)<<3);
		//occ[i*DFACTOR+DFACTOR-1] = t2-t;
		occ[i*DFACTOR] = t2;
	}
	__syncthreads();

	int buffer_p = occ[SM_SIZE00-2];
	// init stream buffer
	for(i=threadIdx.x;i<np;i+=blockDim.x) { //actually, 1*np size is enough
		if(occ[DFACTOR*i] >=0) {
			pb2[blockIdx.x+np*i] = occ[DFACTOR*i]; //transposed
		} else {
			pb2[blockIdx.x+np*i] = PSIZE;
			occ[DFACTOR*i] = 0;
		}		
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		occ[SM_SIZE00-2] = 0;
	}

	//prefix-sum for pb1 ( 8 chunks)

    int sync_upper_np = upper_np;
    if((upper_np&512) > 0) sync_upper_np += 512;

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("(%d %d)\n", upper_np, sync_upper_np);

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
	int d = (np_base>>1);
	int offset=1;
	for(; d > 0; d>>=1) {
		__syncthreads();
		j=(i&511);
		if(j < d && i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				occ[a0+bi] += occ[a0+ai];
//			}
		}
		offset *= 2;
	}
    }
	__syncthreads();

	if(threadIdx.x < (upper_np>>9)) {
//		if(threadIdx.x == 0) es[blockIdx.x] = occ[-1];
		buff[blockIdx.x*32+threadIdx.x] = occ[(threadIdx.x+1)*np_base-1];
		occ[(threadIdx.x+1)*np_base-1] = 0;
	}

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
		int offset = 512;
	for(int d=1; d<np_base; d*=2) {
		offset >>= 1;
		__syncthreads();
		j=(i&511);
		if(j < d & i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				int dummy = occ[a0+ai];
				occ[a0+ai] = occ[a0+bi];
				occ[a0+bi] += dummy; 
//			}
			}
	}
    }
	__syncthreads();
	int base_value = 0;
	for(i=1;i<(upper_np>>9);i++) {
		base_value += buff[blockIdx.x*32+i-1];
		for(j=threadIdx.x;j<np_base;j+=blockDim.x) {
			occ[i*np_base+j] += base_value;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		es[blockIdx.x] = occ[np_base*i-1];
	}
	__syncthreads();

	for(i=threadIdx.x; i<np; i+=blockDim.x) {
		pb1[blockIdx.x*np+i] = occ[i*DFACTOR];
	}	
}




__global__ void
preprocessing_step1big(int DFACTOR, int LOG_DFACTOR, int nv, int ne, int np, int upper_np, int *csc_v, const int * __restrict__ csc_e, int *es, int *pb1, int *pb2,
int *dcnt, int *dindex, int *dx, int *dy)
{

	__shared__ int occ[SM_SIZE0];
//	__shared__ int buffer1[BG], buffer3[BG];
//	__shared__ int buffer_p[2];

	int i, j, k, index, index_size, bias;

//	int dd=0, d0=0, d1=0, d2=0, d3=0;
//	int d4=0, d5=0, d6=0, d7=0;
	int dd=0;

	// get occurrance
	for(i=threadIdx.x;i<upper_np;i+=blockDim.x) {
		occ[i] = 0;
	}

	__syncthreads();
	for(index=blockIdx.x*PSIZE+(threadIdx.x>>0); index-blockIdx.x*PSIZE < PSIZE; ) {
//		__syncthreads();
		for(k=0;k<1;k++) {
			if(index < nv) {
				index_size = csc_v[index+1] - csc_v[index];
				// step 1 (small granularity)
				for(i=0; i<index_size; i++) {
//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("etype1 %d %d\n", index, csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("etype2 %d %d\n", index, csc_e[csc_v[index]+i]);
					int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE0) printf("etype3 : %d %d\n", index, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
					int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//atomicPSeg(&occ[index_pnt], index_pnt);
					if(index_pnt != blockIdx.x) {
						atomicPSeg(&occ[ti], ti);
					} else {
						dd++;
					}
	

				//occ[index_pnt]++;
				}
			}
			index += 1024;
		}
		//__syncthreads();
	}

        dd += __shfl_down(dd, 16);
        dd += __shfl_down(dd, 8);
        dd += __shfl_down(dd, 4);
        dd += __shfl_down(dd, 2);
        dd += __shfl_down(dd, 1);

        if((threadIdx.x&31)==0) {
                occ[SM_SIZE0-3-(threadIdx.x>>5)] = dd;
        }
        __syncthreads();
        if(threadIdx.x<32) {
                dd = occ[SM_SIZE0-3-threadIdx.x];
                dd += __shfl_down(dd, 16);
                dd += __shfl_down(dd, 8);
                dd += __shfl_down(dd, 4);
                dd += __shfl_down(dd, 2);
                dd += __shfl_down(dd, 1);
        }
        __syncthreads();
	
	if(threadIdx.x == 0) {
		occ[DFACTOR*blockIdx.x] = dd;
	}

	if(threadIdx.x < 2) {
		occ[SM_SIZE0-2+threadIdx.x] = 0;
	}
	__syncthreads();

	//make dense-information, normalization
	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		int t=0;
		for(j=0;j<DFACTOR-1;j++) {
			t += occ[i*DFACTOR+j];
			occ[i*DFACTOR+j] = 0;
		}
		int t2=t+occ[i*DFACTOR+DFACTOR-1];
		occ[i*DFACTOR+DFACTOR-1] = 0;
		t2 = (((t2+7)>>3)<<3);
		//occ[i*DFACTOR+DFACTOR-1] = t2-t;
		occ[i*DFACTOR] = t2;
	}
	__syncthreads();

	int buffer_p = occ[SM_SIZE0-2];
	// init stream buffer
	for(i=threadIdx.x;i<np;i+=blockDim.x) { //actually, 1*np size is enough
		if(occ[DFACTOR*i] >=0) {
			pb2[blockIdx.x+np*i] = occ[DFACTOR*i]; //transposed
		} else {
			pb2[blockIdx.x+np*i] = PSIZE;
			occ[DFACTOR*i] = 0;
		}		
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		occ[SM_SIZE0-2] = 0;
	}

	//prefix-sum for pb1 ( 8 chunks)

    int sync_upper_np = upper_np;
    if((upper_np&512) > 0) sync_upper_np += 512;

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("(%d %d)\n", upper_np, sync_upper_np);

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
	int d = (np_base>>1);
	int offset=1;
	for(; d > 0; d>>=1) {
		__syncthreads();
		j=(i&511);
		if(j < d && i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				occ[a0+bi] += occ[a0+ai];
//			}
		}
		offset *= 2;
	}
    }
	__syncthreads();

	if(threadIdx.x < (upper_np>>9)) {
//		if(threadIdx.x == 0) es[blockIdx.x] = occ[-1];
		occ[upper_np+threadIdx.x] = occ[(threadIdx.x+1)*np_base-1];
		occ[(threadIdx.x+1)*np_base-1] = 0;
	}

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
		int offset = 512;
	for(int d=1; d<np_base; d*=2) {
		offset >>= 1;
		__syncthreads();
		j=(i&511);
		if(j < d & i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				int dummy = occ[a0+ai];
				occ[a0+ai] = occ[a0+bi];
				occ[a0+bi] += dummy; 
//			}
			}
	}
    }
	__syncthreads();
	int base_value = 0;
	for(i=1;i<(upper_np>>9);i++) {
		base_value += occ[upper_np+i-1];
		for(j=threadIdx.x;j<np_base;j+=blockDim.x) {
			occ[i*np_base+j] += base_value;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		es[blockIdx.x] = occ[np_base*i-1];
	}
	__syncthreads();

	for(i=threadIdx.x; i<np; i+=blockDim.x) {
		pb1[blockIdx.x*np+i] = occ[i*DFACTOR];
	}	
}




__global__ void
preprocessing_step1medium(int DFACTOR, int LOG_DFACTOR, int nv, int ne, int np, int upper_np, int *csc_v, const int * __restrict__ csc_e, int *es, int *pb1, int *pb2,
int *dcnt, int *dindex, int *dx, int *dy)
{

	__shared__ int occ[SM_SIZE0];
//	__shared__ int buffer1[BG], buffer3[BG];
//	__shared__ int buffer_p[2];

	int i, j, k, index, index_size, bias;

//	int dd=0, d0=0, d1=0, d2=0, d3=0;
//	int d4=0, d5=0, d6=0, d7=0;
	int dd=0;

	// get occurrance
	for(i=threadIdx.x;i<upper_np;i+=blockDim.x) {
		occ[i] = 0;
	}

	__syncthreads();
	for(index=blockIdx.x*PSIZE+(threadIdx.x>>2); index-blockIdx.x*PSIZE < PSIZE; ) {
//		__syncthreads();
		for(k=0;k<4;k++) {
			if(index < nv) {
				index_size = csc_v[index+1] - csc_v[index];
				// step 1 (small granularity)
				for(i=(threadIdx.x&3); i<index_size; i+=4) {
//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("etype1 %d %d\n", index, csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("etype2 %d %d\n", index, csc_e[csc_v[index]+i]);
					int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE0) printf("etype3 : %d %d\n", index, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
					int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//atomicPSeg(&occ[index_pnt], index_pnt);
					if(index_pnt != blockIdx.x) {
						atomicPSeg(&occ[ti], ti);
					} else {
						dd++;
					}
	

				//occ[index_pnt]++;
				}
			}
			index += 256;
		}
		//__syncthreads();
	}

        dd += __shfl_down(dd, 16);
        dd += __shfl_down(dd, 8);
        dd += __shfl_down(dd, 4);
        dd += __shfl_down(dd, 2);
        dd += __shfl_down(dd, 1);

        if((threadIdx.x&31)==0) {
                occ[SM_SIZE0-3-(threadIdx.x>>5)] = dd;
        }
        __syncthreads();
        if(threadIdx.x<32) {
                dd = occ[SM_SIZE0-3-threadIdx.x];
                dd += __shfl_down(dd, 16);
                dd += __shfl_down(dd, 8);
                dd += __shfl_down(dd, 4);
                dd += __shfl_down(dd, 2);
                dd += __shfl_down(dd, 1);
        }
        __syncthreads();
	
	if(threadIdx.x == 0) {
		occ[DFACTOR*blockIdx.x] = dd;
	}

	if(threadIdx.x < 2) {
		occ[SM_SIZE0-2+threadIdx.x] = 0;
	}
	__syncthreads();

	//make dense-information, normalization
	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		int t=0;
		for(j=0;j<DFACTOR-1;j++) {
			t += occ[i*DFACTOR+j];
			occ[i*DFACTOR+j] = 0;
		}
		int t2=t+occ[i*DFACTOR+DFACTOR-1];
		occ[i*DFACTOR+DFACTOR-1] = 0;
		t2 = (((t2+7)>>3)<<3);
		//occ[i*DFACTOR+DFACTOR-1] = t2-t;
		occ[i*DFACTOR] = t2;
	}
	__syncthreads();

	int buffer_p = occ[SM_SIZE0-2];
	// init stream buffer
	for(i=threadIdx.x;i<np;i+=blockDim.x) { //actually, 1*np size is enough
		if(occ[DFACTOR*i] >=0) {
			pb2[blockIdx.x+np*i] = occ[DFACTOR*i]; //transposed
		} else {
			pb2[blockIdx.x+np*i] = PSIZE;
			occ[DFACTOR*i] = 0;
		}		
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		occ[SM_SIZE0-2] = 0;
	}

	//prefix-sum for pb1 ( 8 chunks)

    int sync_upper_np = upper_np;
    if((upper_np&512) > 0) sync_upper_np += 512;

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("(%d %d)\n", upper_np, sync_upper_np);

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
	int d = (np_base>>1);
	int offset=1;
	for(; d > 0; d>>=1) {
		__syncthreads();
		j=(i&511);
		if(j < d && i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				occ[a0+bi] += occ[a0+ai];
//			}
		}
		offset *= 2;
	}
    }
	__syncthreads();

	if(threadIdx.x < (upper_np>>9)) {
//		if(threadIdx.x == 0) es[blockIdx.x] = occ[-1];
		occ[upper_np+threadIdx.x] = occ[(threadIdx.x+1)*np_base-1];
		occ[(threadIdx.x+1)*np_base-1] = 0;
	}

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
		int offset = 512;
	for(int d=1; d<np_base; d*=2) {
		offset >>= 1;
		__syncthreads();
		j=(i&511);
		if(j < d & i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				int dummy = occ[a0+ai];
				occ[a0+ai] = occ[a0+bi];
				occ[a0+bi] += dummy; 
//			}
			}
	}
    }
	__syncthreads();
	int base_value = 0;
	for(i=1;i<(upper_np>>9);i++) {
		base_value += occ[upper_np+i-1];
		for(j=threadIdx.x;j<np_base;j+=blockDim.x) {
			occ[i*np_base+j] += base_value;
		}
		__syncthreads();
	}
	if(threadIdx.x == 0) {
		es[blockIdx.x] = occ[np_base*i-1];
	}
	__syncthreads();

	for(i=threadIdx.x; i<np; i+=blockDim.x) {
		pb1[blockIdx.x*np+i] = occ[i*DFACTOR];
	}	
}








__global__ void
__launch_bounds__(BSIZE, 1)
preprocessing_step1(int DFACTOR, int LOG_DFACTOR, int nv, int ne, int np, int upper_np, int *csc_v, int *csc_e, int *es, int *pb1, int *pb2,
int *dcnt, int *dindex, int *dx, int *dy)
{

	__shared__ int occ[SM_SIZE];
	__shared__ short buffer1[BG], buffer3[BG];
//	__shared__ int buffer_p[2];

	int i, j, k, index, index_size, bias;

//	int dd=0, d0=0, d1=0, d2=0, d3=0;
//	int d4=0, d5=0, d6=0, d7=0;

	// get occurrance
	for(i=threadIdx.x;i</*upper_np*/SM_SIZE;i+=blockDim.x) {
		occ[i] = 0;
	}

	//__syncthreads();
	for(index=blockIdx.x*PSIZE+(threadIdx.x>>2); index-blockIdx.x*PSIZE < PSIZE; ) {
		if(threadIdx.x < 2) {
			occ[SM_SIZE-2+threadIdx.x] = 0;
		}
		__syncthreads();
		for(k=0;k<4;k++) {
			if(index < nv) {
				index_size = csc_v[index+1] - csc_v[index];
				if(index_size >= 32) {
					bias = index_size - (index_size&31);
					if((threadIdx.x&3) == 0) {
						int p = atomicAggInc(&occ[SM_SIZE-2]);
						buffer1[p] = index - blockIdx.x*PSIZE;
					}
					if(index_size >= 256) {
						if((threadIdx.x&3) == 0) {
							int p2 = atomicAggInc(&occ[SM_SIZE-1]);
							buffer3[p2] = index - blockIdx.x*PSIZE;
						}
					}
				} else {
					bias = 0;
				}
				// step 1 (small granularity)
				for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("etype1 %d %d\n", index, csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("etype2 %d %d\n", index, csc_e[csc_v[index]+i]);
					int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE) printf("etype3 : %d %d\n", index, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
					int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//atomicPSeg(&occ[index_pnt], index_pnt);
					atomicPSeg(&occ[ti], ti);

/*
					if(index_pnt == blockIdx.x) {
						dd++;
					} else if(index_pnt < 8) {
						if(index_pnt == 0) d0++;
						else if(index_pnt == 1) d1++;
						else if(index_pnt == 2) d2++;
						else if(index_pnt == 3) d3++;
						else if(index_pnt == 4) d4++;
						else if(index_pnt == 5) d5++;
						else if(index_pnt == 6) d6++;
						else if(index_pnt == 7) d7++;
					} else {
						int ix = DFACTOR*index_pnt+((threadIdx.x>>5)&(DFACTOR-1));
						atomicSeg(&occ[ix], ix);
						int tr = __shfl(index_pnt, 0);
						int tr2 = __shfl(index_pnt, 16);
						if(tr == index_pnt || tr2 == index_pnt) {
							if(tr == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+((threadIdx.x>>5)&(DFACTOR-1))]);
							if(tr != tr2 && tr2 == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+((threadIdx.x>>5)&(DFACTOR-1))]);
						} else {
							atomicAdd(&occ[DFACTOR*index_pnt+((threadIdx.x>>5)&(DFACTOR-1))], 1);
						}
					}*/



				//occ[index_pnt]++;
				}
			}
			index += 256;
		}
		__syncthreads();
		// step 2 (medium granularity)
		for(i=(threadIdx.x>>5);i<occ[SM_SIZE-2];i+=32) {
			int lindex = buffer1[i]+blockIdx.x*PSIZE;
			int dummy = csc_v[lindex+1] - csc_v[lindex];
			int bf2 = dummy - (dummy&31);
			int bf22 = bf2 - (bf2&255);
			for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
//if(csc_v[lindex]+j < 0 || csc_v[lindex]+j >= ne) printf("etype2-1 %d %d\n", lindex, csc_v[lindex]+j);
//if(csc_e[csc_v[lindex]+j] < 0 || csc_e[csc_v[lindex]+j] >= nv) printf("etype2-2 %d %d\n", lindex, csc_e[csc_v[lindex]+j]);
				int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
				int ti = index_pnt * DFACTOR + (lindex & (DFACTOR-1));
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE) printf("etype3 : %d %d\n", lindex, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
				//atomicSeg(&occ[index_pnt], index_pnt);
				atomicSeg(&occ[ti], ti);
				/*
				if(index_pnt == blockIdx.x) {
					dd++;
				} else if(index_pnt < 8) {
					if(index_pnt == 0) d0++;
					else if(index_pnt == 1) d1++;
					else if(index_pnt == 2) d2++;
					else if(index_pnt == 3) d3++;
					else if(index_pnt == 4) d4++;
					else if(index_pnt == 5) d5++;
					else if(index_pnt == 6) d6++;
					else if(index_pnt == 7) d7++;
				} else {
					int ix = DFACTOR*index_pnt+(lindex&(DFACTOR-1));
					atomicSeg(&occ[ix], ix);
					
					int tr = __shfl(index_pnt, 0);
					int tr2 = __shfl(index_pnt, 16);
					if(tr == index_pnt || tr2 == index_pnt) {
						if(tr == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+(lindex&(DFACTOR-1))]);
						if(tr != tr2 && tr2 == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+(lindex&(DFACTOR-1))]);
					} else {
						atomicAdd(&occ[DFACTOR*index_pnt+(lindex&(DFACTOR-1))], 1);
					}

					//atomicAdd(&occ[DFACTOR*index_pnt+(lindex&(DFACTOR-1))], 1);
				}*/
				//occ[index_pnt]++;
			}
		}
		// step 3 (large granularity)
		for(i=(threadIdx.x>>8);i<occ[SM_SIZE-1];i+=4) {
			int lindex = buffer3[i]+blockIdx.x*PSIZE;
			int h = ((threadIdx.x>>5)&7);
			int bf2 = csc_v[lindex+1] - csc_v[lindex];
			bf2 = bf2 - (bf2&255);
			for(j=(threadIdx.x&255);j<bf2;j+=256) {
//if(csc_v[lindex]+j < 0 || csc_v[lindex]+j >= ne) printf("etype3-1 %d %d\n", lindex, csc_v[lindex]+j);
//if(csc_e[csc_v[lindex]+j] < 0 || csc_e[csc_v[lindex]+j] >= nv) printf("etype3-2 %d %d\n", lindex, csc_e[csc_v[lindex]+j]);
				int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE) printf("etype3 : %d %d\n", lindex, DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)));
				int ti = index_pnt * DFACTOR + (h & (DFACTOR-1));
				//atomicSeg(&occ[index_pnt], index_pnt);
				atomicSeg(&occ[ti], ti);
				
				/*
				if(index_pnt == blockIdx.x) {
					dd++;
				} else if(index_pnt < 8) {
					if(index_pnt == 0) d0++;
					else if(index_pnt == 1) d1++;
					else if(index_pnt == 2) d2++;
					else if(index_pnt == 3) d3++;
					else if(index_pnt == 4) d4++;
					else if(index_pnt == 5) d5++;
					else if(index_pnt == 6) d6++;
					else if(index_pnt == 7) d7++;
				} else {
//					atomicAdd(&occ[DFACTOR*index_pnt+(lindex&(DFACTOR-1))], 1);
					int ix = DFACTOR*index_pnt+((h)&(DFACTOR-1));
					atomicSeg(&occ[ix], ix);	
					
					int tr = __shfl(index_pnt, 0);
					int tr2 = __shfl(index_pnt, 16);
					if(tr == index_pnt || tr2 == index_pnt) {
						if(tr == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+((h)&(DFACTOR-1))]);
						if(tr != tr2 && tr2 == index_pnt) atomicAggInc(&occ[DFACTOR*index_pnt+((h)&(DFACTOR-1))]);
					} else {
						atomicAdd(&occ[DFACTOR*index_pnt+((h)&(DFACTOR-1))], 1);
					}
				}*/

				//occ[index_pnt]++;
			}
		}
		__syncthreads();
	}

//if(threadIdx.x == 0)
//	for(i=0;i<4;i++) printf("%d\n", occ[i]);

/*
	//reduction
	for(j=(DFACTOR>>1);j>=1;j>>=1) {
		for(i=threadIdx.x;i<np*j;i+=blockDim.x) {
			occ[i] += occ[i+j*np];
		}
		if(j > 1) { // safe
			 __syncthreads();
		}
	}
*/

	if(threadIdx.x < 2) {
		occ[SM_SIZE-2+threadIdx.x] = 0;
	}
	__syncthreads();
	//make dense-information, normalization
	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		int t=0;
		for(j=0;j<DFACTOR-1;j++) {
			t += occ[i*DFACTOR+j];
			occ[i*DFACTOR+j] = 0;
		}
		int t2=t+occ[i*DFACTOR+DFACTOR-1];
		occ[i*DFACTOR+DFACTOR-1] = 0;
		t2 = (((t2+7)>>3)<<3);
		//occ[i*DFACTOR+DFACTOR-1] = t2-t;
		if(t2 >= THRESHOLD) {
			int k = atomicAdd(dcnt, 1);
			t2 = ((((t2+BSIZE-1)>>LOG_BSIZE)<<LOG_BSIZE)+PSIZE+BSIZE);// + PSIZE*3;
			//d_check[np*blockIdx.x+i] = k;
			dx[k] = blockIdx.x;
			dy[k] = i;
			//for(j=0;j<1;j++) {
			//	if(j < DFACTOR-1) dindex[k+j] = occ[i*DFACTOR+j];
			//	else dindex[k+j] = t2-t;
				dindex[k] = t2;
				occ[i*DFACTOR] = -1;
			//}
			int k2 = atomicAdd(&occ[SM_SIZE-2], 1);
			buffer1[k2] = i; // loc
			buffer3[k2] = k; // value
		}
		else occ[i*DFACTOR] = t2;
	}
	__syncthreads();

	int buffer_p = occ[SM_SIZE-2];
	// init stream buffer
	for(i=threadIdx.x;i<np;i+=blockDim.x) { //actually, 1*np size is enough
		if(occ[DFACTOR*i] >=0) {
			pb2[blockIdx.x+np*i] = occ[DFACTOR*i]; //transposed
		} else {
			pb2[blockIdx.x+np*i] = PSIZE;
			occ[DFACTOR*i] = 0;
		}		
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		occ[SM_SIZE-2] = 0;
	}

/*
__syncthreads();
if(blockIdx.x == 0 && threadIdx.x == 0) {
	for(i=0;i<1025;i++) {
		printf("%d ", occ[i]);
	}
	printf("\nstep1\n\n");
}
__syncthreads();*/

	//prefix-sum for pb1 ( 8 chunks)

    int sync_upper_np = upper_np;
    if((upper_np&512) > 0) sync_upper_np += 512;

//if(threadIdx.x == 0 && blockIdx.x == 0) printf("(%d %d)\n", upper_np, sync_upper_np);

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
	int d = (np_base>>1);
	int offset=1;
	for(; d > 0; d>>=1) {
		__syncthreads();
		j=(i&511);
		if(j < d && i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				occ[a0+bi] += occ[a0+ai];
//			}
		}
		offset *= 2;
	}
    }
	__syncthreads();

	if(threadIdx.x < (upper_np>>9)) {
//		if(threadIdx.x == 0) es[blockIdx.x] = occ[-1];
		occ[upper_np+threadIdx.x] = occ[(threadIdx.x+1)*np_base-1];
		occ[(threadIdx.x+1)*np_base-1] = 0;
	}

    for(i=threadIdx.x;i<sync_upper_np;i+=blockDim.x) {
		int offset = 512;
	for(int d=1; d<np_base; d*=2) {
		offset >>= 1;
		__syncthreads();
		j=(i&511);
		if(j < d & i<upper_np) {
			int a0 = (i>>9)*np_base;
			int ai = offset*(2*j+1)-1;
			int bi = offset*(2*j+2)-1;
//			if(ai < np*DFACTOR && bi < np*DFACTOR) {
				int dummy = occ[a0+ai];
				occ[a0+ai] = occ[a0+bi];
				occ[a0+bi] += dummy; 
//			}
			}
	}
    }
	__syncthreads();
	int base_value = 0;
	for(i=1;i<(upper_np>>9);i++) {
		base_value += occ[upper_np+i-1];
		for(j=threadIdx.x;j<np_base;j+=blockDim.x) {
			occ[i*np_base+j] += base_value;
		}
		__syncthreads();
	}
	__syncthreads();
	if(threadIdx.x == 0) {
		es[blockIdx.x] = occ[np_base*i-1];
	}
	__syncthreads();


/*
__syncthreads();
if(blockIdx.x == 0 && threadIdx.x == 0) {
	for(i=0;i<1025;i++) {
		printf("%d ", occ[i]);
	}
	printf("\nstep2\n\n");
}
__syncthreads();
return;*/

/*
if(blockIdx.x == 0 && threadIdx.x == 0) {
	for(i=0;i<np*DFACTOR; i++)
		printf("(%d) ", occ[i]);
	printf("\n"); 
} __syncthreads();
*/
	for(i=threadIdx.x; i<buffer_p; i+=blockDim.x) {
		occ[DFACTOR*buffer1[i]] = -(buffer3[i]+1);
	}
	__syncthreads();

	for(i=threadIdx.x; i<np; i+=blockDim.x) {
		pb1[blockIdx.x*np+i] = occ[i*DFACTOR];
	}	


}

__global__ void	preprocessing_step11(int np, int DFACTOR, int *cpb2, int *cey)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
	if(index <= np)
		cey[index] = cpb2[np*DFACTOR*index];
}




__global__ void
preprocessing_step2verybig(int DFACTOR, int nv, int ne, int upper_ne, int np, int upper_np, int *csc_v, int *csc_e, const int * __restrict__ pb1, const int * __restrict__ pb2, int *es,
int dcnt, int *dindex, int *dx, int *dy, short *sp1, short *sp2, short *fx, short *fy, int *mapper
#ifdef E1
, E1T *csc_ev, E1T *sp1v, E1T *fz
#endif
)
{
	__shared__ int occ[SM_SIZE00];
//        __shared__ int buffer1[BG], buffer3[BG];
//        __shared__ int buffer_p[2];

//	__shared__ int spb1[SM_SIZE00];
//	__shared__ int spb2[SM_SIZE00];

	int i, j, index, index_size, bias;

	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		occ[i] = 0; 
//		spb1[i] = pb1[np*DFACTOR*blockIdx.x+i];
//		spb2[i] = pb2[np*DFACTOR*blockIdx.x+i];
	}
	__syncthreads();

        for(index=blockIdx.x*PSIZE+(threadIdx.x>>0); index-blockIdx.x*PSIZE < PSIZE; ) {
//                if(threadIdx.x < 2) {
//                        occ[SM_SIZE00-2+threadIdx.x] = 0;
//                }
//                __syncthreads();
                for(int lk=0;lk<1;lk++) {
                        if(index < nv) {
                                index_size = csc_v[index+1] - csc_v[index];
                                // step 1 (small granularity)
                                for(i=0; i<index_size; i++) {
                                        int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
					int k;
					//int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//k = atomicPSeg(&occ[ti], ti);
					if(index_pnt != blockIdx.x) {
						k = atomicPSeg(&occ[index_pnt], index_pnt);
					} 
					if(index_pnt == blockIdx.x) {
						k = atomicAggInc(&occ[index_pnt]);
					}
					int flag = pb1[blockIdx.x*np+index_pnt];
					int dst_v0 = csc_v[index]+i;
					int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v < 0 || dst_v >= PSIZE) printf("e : %d\n", dst_v);
					if(flag >= 0) { //sparse

						flag += es[blockIdx.x];
//if(k+flag < 0 || k+flag >= upper_ne) printf("bound1 %d (%d %d %d %d) %d\n",k, pb1[0],pb1[1],pb1[2],pb1[3], upper_ne);
						sp1[k+flag] = index - blockIdx.x*PSIZE; //src
#ifdef E1
						sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 100;
//printf("%d %d %d\n", k+flag, dst_v0, csc_ev[dst_v0]);
#endif
						sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst

						if(((k+flag)&7) == 0) {
							mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
						}
					}
                                }
                        }
                        index += 1024;
		}
        }
}




__global__ void
preprocessing_step2big(int DFACTOR, int nv, int ne, int upper_ne, int np, int upper_np, int *csc_v, int *csc_e, const int * __restrict__ pb1, const int * __restrict__ pb2, int *es,
int dcnt, int *dindex, int *dx, int *dy, short *sp1, short *sp2, short *fx, short *fy, int *mapper
#ifdef E1
, E1T *csc_ev, E1T *sp1v, E1T *fz
#endif
)
{
	__shared__ int occ[SM_SIZE0];
//        __shared__ int buffer1[BG], buffer3[BG];
//        __shared__ int buffer_p[2];

//	__shared__ int spb1[SM_SIZE0];
//	__shared__ int spb2[SM_SIZE0];

	int i, j, index, index_size, bias;

	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		occ[i] = 0; 
//		spb1[i] = pb1[np*DFACTOR*blockIdx.x+i];
//		spb2[i] = pb2[np*DFACTOR*blockIdx.x+i];
	}
	__syncthreads();

        for(index=blockIdx.x*PSIZE+(threadIdx.x>>0); index-blockIdx.x*PSIZE < PSIZE; ) {
//                if(threadIdx.x < 2) {
//                        occ[SM_SIZE0-2+threadIdx.x] = 0;
//                }
//                __syncthreads();
                for(int lk=0;lk<1;lk++) {
                        if(index < nv) {
                                index_size = csc_v[index+1] - csc_v[index];
                                // step 1 (small granularity)
                                for(i=0; i<index_size; i++) {
                                        int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
					int k;
					//int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//k = atomicPSeg(&occ[ti], ti);
					if(index_pnt != blockIdx.x) {
						k = atomicPSeg(&occ[index_pnt], index_pnt);
					} 
					if(index_pnt == blockIdx.x) {
						k = atomicAggInc(&occ[index_pnt]);
					}
					int flag = pb1[blockIdx.x*np+index_pnt];
					int dst_v0 = csc_v[index]+i;
					int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v < 0 || dst_v >= PSIZE) printf("e : %d\n", dst_v);
					if(flag >= 0) { //sparse

						flag += es[blockIdx.x];
//if(k+flag < 0 || k+flag >= upper_ne) printf("bound1 %d (%d %d %d %d) %d\n",k, pb1[0],pb1[1],pb1[2],pb1[3], upper_ne);
						sp1[k+flag] = index - blockIdx.x*PSIZE; //src
#ifdef E1
						sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 100;
//printf("%d %d %d\n", k+flag, dst_v0, csc_ev[dst_v0]);
#endif
						sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst

						if(((k+flag)&7) == 0) {
							mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
						}
					}
                                }
                        }
                        index += 1024;
		}
        }
}


__global__ void
__launch_bounds__(BSIZE, 1)
preprocessing_step2medium(int DFACTOR, int nv, int ne, int upper_ne, int np, int upper_np, int *csc_v, int *csc_e, const int * __restrict__ pb1, const int * __restrict__ pb2, int *es,
int dcnt, int *dindex, int *dx, int *dy, short *sp1, short *sp2, short *fx, short *fy, int *mapper
#ifdef E1
, E1T *csc_ev, E1T *sp1v, E1T *fz
#endif
)
{
	__shared__ int occ[SM_SIZE0];
//        __shared__ int buffer1[BG], buffer3[BG];
//        __shared__ int buffer_p[2];

//	__shared__ int spb1[SM_SIZE0];
//	__shared__ int spb2[SM_SIZE0];

	int i, j, index, index_size, bias;

	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		occ[i] = 0; 
//		spb1[i] = pb1[np*DFACTOR*blockIdx.x+i];
//		spb2[i] = pb2[np*DFACTOR*blockIdx.x+i];
	}
	__syncthreads();

        for(index=blockIdx.x*PSIZE+(threadIdx.x>>2); index-blockIdx.x*PSIZE < PSIZE; ) {
//                if(threadIdx.x < 2) {
//                        occ[SM_SIZE0-2+threadIdx.x] = 0;
//                }
//                __syncthreads();
                for(int lk=0;lk<4;lk++) {
                        if(index < nv) {
                                index_size = csc_v[index+1] - csc_v[index];
                                // step 1 (small granularity)
                                for(i=(threadIdx.x&3); i<index_size; i+=4) {
                                        int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
					int k;
					//int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//k = atomicPSeg(&occ[ti], ti);
					if(index_pnt != blockIdx.x) {
						k = atomicPSeg(&occ[index_pnt], index_pnt);
					} 
					if(index_pnt == blockIdx.x) {
						k = atomicAggInc(&occ[index_pnt]);
					}
					int flag = pb1[blockIdx.x*np+index_pnt];
					int dst_v0 = csc_v[index]+i;
					int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v < 0 || dst_v >= PSIZE) printf("e : %d\n", dst_v);
					if(flag >= 0) { //sparse

						flag += es[blockIdx.x];
//if(k+flag < 0 || k+flag >= upper_ne) printf("bound1 %d (%d %d %d %d) %d\n",k, pb1[0],pb1[1],pb1[2],pb1[3], upper_ne);
						sp1[k+flag] = index - blockIdx.x*PSIZE; //src
#ifdef E1
						sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 100;
//printf("%d %d %d\n", k+flag, dst_v0, csc_ev[dst_v0]);
#endif
						sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst

						if(((k+flag)&7) == 0) {
							mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
						}
					}
                                }
                        }
                        index += 256;
		}
        }
}









__global__ void
__launch_bounds__(BSIZE, 1)
preprocessing_step2(int DFACTOR, int nv, int ne, int upper_ne, int np, int upper_np, int *csc_v, int *csc_e, const int * __restrict__ pb1, const int * __restrict__ pb2, int *es,
int dcnt, int *dindex, int *dx, int *dy, short *sp1, short *sp2, short *fx, short *fy, int *mapper
#ifdef E1
, E1T *csc_ev, E1T *sp1v, E1T *fz
#endif
)
{
	__shared__ int occ[SM_SIZE];
        __shared__ short buffer1[BG], buffer3[BG];
//        __shared__ int buffer_p[2];

//	__shared__ int spb1[SM_SIZE];
//	__shared__ int spb2[SM_SIZE];

	int i, j, index, index_size, bias;

	for(i=threadIdx.x;i<np;i+=blockDim.x) {
		occ[i] = 0; 
//		spb1[i] = pb1[np*DFACTOR*blockIdx.x+i];
//		spb2[i] = pb2[np*DFACTOR*blockIdx.x+i];
	}

        for(index=blockIdx.x*PSIZE+(threadIdx.x>>2); index-blockIdx.x*PSIZE < PSIZE; ) {
                if(threadIdx.x < 2) {
                        occ[SM_SIZE-2+threadIdx.x] = 0;
                }
                __syncthreads();
                for(int lk=0;lk<4;lk++) {
                        if(index < nv) {
                                index_size = csc_v[index+1] - csc_v[index];
                                if(index_size >= 32) {
                                        bias = index_size - (index_size&31);
                                        if((threadIdx.x&3) == 0) {
                                                int p = atomicAggInc(&occ[SM_SIZE-2]);
                                                buffer1[p] = index - blockIdx.x*PSIZE;
                                        }
                                        if(index_size >= 256) {
                                                if((threadIdx.x&3) == 0) {
                                                        int p2 = atomicAggInc(&occ[SM_SIZE-1]);
                                                        buffer3[p2] = index - blockIdx.x*PSIZE;
                                                }
                                        }
                                } else {
                                        bias = 0;
                                }
                                // step 1 (small granularity)
                                for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                                        int index_pnt = csc_e[csc_v[index]+i] / PSIZE;
					int k;
//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("err0 %d \n", csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("err %d \n", csc_e[csc_v[index]+i]);
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE) printf("err2\n");
					//int ti = index_pnt * DFACTOR + ((threadIdx.x>>5)&(DFACTOR-1));
					//k = atomicPSeg(&occ[ti], ti);
					k = atomicPSeg(&occ[index_pnt], index_pnt);

					int flag = pb1[blockIdx.x*np+index_pnt];
					int dst_v0 = csc_v[index]+i;
					int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v < 0 || dst_v >= PSIZE) printf("e : %d\n", dst_v);
					if(flag >= 0) { //sparse

						flag += es[blockIdx.x];
//if(k+flag < 0 || k+flag >= upper_ne) printf("bound1 %d (%d %d %d %d) %d\n",k, pb1[0],pb1[1],pb1[2],pb1[3], upper_ne);
						sp1[k+flag] = index - blockIdx.x*PSIZE; //src
#ifdef E1
						sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 100;
//printf("%d %d %d\n", k+flag, dst_v0, csc_ev[dst_v0]);
#endif
//if(index - blockIdx.x*PSIZE < 0 || index - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", index);
//if(k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))] < 0 
//|| k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))] >= upper_ne)
//printf("bound2 %d %d\n", k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))], upper_ne); 
						sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst

						if(((k+flag)&7) == 0) {
							mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
						}
					} else { //dense
//printf("er\n");
						flag = -(flag+1);
// printf("%d %d %d\n", dcnt, flag, dindex[flag]+k);
						fx[dindex[flag]+k] = index - blockIdx.x*PSIZE; //src
						fy[dindex[flag]+k] = dst_v;//dst
#ifdef E1
						fz[dindex[flag]+k] = csc_ev[dst_v0];//dst
#endif
					}
                                }
                        }
                        index += 256;
                }
                __syncthreads();
                // step 2 (medium granularity)
                for(i=(threadIdx.x>>5);i<occ[SM_SIZE-2];i+=32) {
                        int lindex = buffer1[i] + blockIdx.x*PSIZE;
                        int dummy = csc_v[lindex+1] - csc_v[lindex];
                        int bf2 = dummy - (dummy&31);
                        int bf22 = bf2 - (bf2&255);
                        for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
                                int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
				//int ti = index_pnt * DFACTOR + (lindex & (DFACTOR-1));
				int k;
				k = atomicSeg(&occ[index_pnt], index_pnt);
				//k = atomicSeg(&occ[ti], ti);

				//int k = atomicAdd(&occ[DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1))],1);
				int flag = pb1[blockIdx.x*np+index_pnt];
				int dst_v0 = csc_v[lindex]+j;
				int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v > PSIZE) printf("e : %d\n", dst_v);
				if(flag >= 0) { //sparse

					flag += es[blockIdx.x];
					sp1[k+flag] = lindex - blockIdx.x*PSIZE; //src
#ifdef E1
					sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 1000;
#endif

//if(lindex - blockIdx.x*PSIZE < 0 || lindex - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", lindex);
					sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst
					if(((k+flag)&7) == 0) {
						mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
					}
				} else { //dense
//printf("er\n");
					flag = -(flag+1);
//if(threadIdx.x==0) printf("%d %d %d\n", dcnt, flag, dindex[flag]+k);
// printf("%d %d %d\n", dcnt, flag, dindex[flag]+k);
					fx[dindex[flag]+k] = lindex - blockIdx.x*PSIZE; //src
					fy[dindex[flag]+k] = dst_v;//dst
#ifdef E1
					fz[dindex[flag]+k] = csc_ev[dst_v0];//dst
#endif
				}
                        }
                }
                // step 3 (large granularity)
                for(i=(threadIdx.x>>8);i<occ[SM_SIZE-1];i+=4) {
                        int lindex = buffer3[i] + blockIdx.x*PSIZE;
			int h = ((threadIdx.x>>5)&7);
                        int bf2 = csc_v[lindex+1] - csc_v[lindex];
                        bf2 = bf2 - (bf2&255);
                        for(j=(threadIdx.x&255);j<bf2;j+=256) {
                                int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
				//int ti = index_pnt * DFACTOR + (h & (DFACTOR-1));
				int k;
				k = atomicSeg(&occ[index_pnt], index_pnt);
				//k = atomicSeg(&occ[ti], ti);

				//int k = atomicAdd(&occ[DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1))],1);
				int flag = pb1[blockIdx.x*np+index_pnt];
				int dst_v0 = csc_v[lindex]+j;
				int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v > PSIZE) printf("e : %d\n", dst_v);
				if(flag >= 0) { //sparse
				
					flag += es[blockIdx.x];
					sp1[k+flag] = lindex - blockIdx.x*PSIZE; //src
#ifdef E1
					sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 10000;
#endif

//if(lindex - blockIdx.x*PSIZE < 0 || lindex - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", lindex);
					sp2[k+pb2[index_pnt*np+blockIdx.x]] = dst_v; //dst
					if(((k+flag)&7) == 0) {
						mapper[(k+flag)>>3] = k+pb2[index_pnt*np+blockIdx.x];
					}
				} else { //dense
//printf("er\n");
					flag = -(flag+1);
//printf("%d %d %d\n", dcnt, flag, dindex[flag]+k);
					fx[dindex[flag]+k] = lindex - blockIdx.x*PSIZE; //src
					fy[dindex[flag]+k] = dst_v;//dst
#ifdef E1
					fz[dindex[flag]+k] = csc_ev[dst_v0];//dst
#endif
				}
                        }
                }
                __syncthreads();
        }
}


/*
__global__ void
//__launch_bounds__(BSIZE, 2)
preprocessing_step22(int DFACTOR, int nv, int ne, int upper_ne, int np, int upper_np, int *csc_v, int *csc_e, const int * __restrict__ pb1, const int * __restrict__ pb2, int *es,
int dcnt, int *dindex, int *dx, int *dy, short *sp1, short *sp2, short *fx, short *fy, int *mapper
#ifdef E1
, E1T *csc_ev, E1T *sp1v, E1T *fz
#endif
)
{
	__shared__ int occ[SM_SIZE];
        __shared__ int buffer1[BG], buffer3[BG];
//        __shared__ int buffer_p[2];

//	__shared__ int spb1[SM_SIZE];
//	__shared__ int spb2[SM_SIZE];

	int i, j, index, index_size, bias;

	for(i=threadIdx.x;i<DFACTOR*np;i+=blockDim.x) {
		occ[i] = 0; 
//		spb1[i] = pb1[np*DFACTOR*blockIdx.x+i];
//		spb2[i] = pb2[np*DFACTOR*blockIdx.x+i];
	}

        for(index=blockIdx.x*PSIZE+(threadIdx.x>>2); index-blockIdx.x*PSIZE < PSIZE; ) {
                if(threadIdx.x < 2) {
                        occ[SM_SIZE-2+threadIdx.x] = 0;
                }
                __syncthreads();
                for(int lk=0;lk<4;lk++) {
                        if(index < nv) {
                                index_size = csc_v[index+1] - csc_v[index];
                                if(index_size >= 32) {
                                        bias = index_size - (index_size&31);
                                        if((threadIdx.x&3) == 0) {
                                                int p = atomicAggInc(&occ[SM_SIZE-2]);
                                                buffer1[p] = index;
                                        }
                                        if(index_size >= 256) {
                                                if((threadIdx.x&3) == 0) {
                                                        int p2 = atomicAggInc(&occ[SM_SIZE-1]);
                                                        buffer3[p2] = index;
                                                }
                                        }
                                } else {
                                        bias = 0;
                                }
                                // step 1 (small granularity)
                                for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                                        int index_pnt = csc_e[csc_v[index]+i] / PSIZE;

//if(csc_v[index]+i < 0 || csc_v[index]+i >= ne) printf("err0 %d \n", csc_v[index]+i);
//if(csc_e[csc_v[index]+i] < 0 || csc_e[csc_v[index]+i] >= nv) printf("err %d \n", csc_e[csc_v[index]+i]);
//if(DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) < 0 || DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1)) >= SM_SIZE) printf("err2\n");
					int k = atomicAdd(&occ[DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1))],1);
					int flag = pb1[blockIdx.x*np*DFACTOR+index_pnt*DFACTOR+(threadIdx.x&(DFACTOR-1))];
					int dst_v0 = csc_v[index]+i;
					int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v < 0 || dst_v >= PSIZE) printf("e : %d\n", dst_v);
					if(flag >= 0) { //sparse

						flag += es[blockIdx.x];
//if(k+flag < 0 || k+flag >= upper_ne) printf("bound1 %d (%d %d %d %d) %d\n",k, pb1[0],pb1[1],pb1[2],pb1[3], upper_ne);
						sp1[k+flag] = index - blockIdx.x*PSIZE; //src
#ifdef E1
						sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 100;
//printf("%d %d %d\n", k+flag, dst_v0, csc_ev[dst_v0]);
#endif
//if(index - blockIdx.x*PSIZE < 0 || index - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", index);
//if(k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))] < 0 
//|| k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))] >= upper_ne)
//printf("bound2 %d %d\n", k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))], upper_ne); 
						sp2[k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))]] = dst_v; //dst

						if(((k+flag)&7) == 0) {
							mapper[(k+flag)>>3] = k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))];
						}
					}
                                }
                        }
                        index += 256;
                }
                __syncthreads();
                // step 2 (medium granularity)
                for(i=(threadIdx.x>>5);i<occ[SM_SIZE-2];i+=32) {
                        int lindex = buffer1[i];
                        int dummy = csc_v[lindex+1] - csc_v[lindex];
                        int bf2 = dummy - (dummy&31);
                        int bf22 = bf2 - (bf2&255);
                        for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
                                int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
				int k = atomicAdd(&occ[DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1))],1);
				int flag = pb1[blockIdx.x*np*DFACTOR+index_pnt*DFACTOR+(threadIdx.x&(DFACTOR-1))];
				int dst_v0 = csc_v[lindex]+j;
				int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v > PSIZE) printf("e : %d\n", dst_v);
				if(flag >= 0) { //sparse

					flag += es[blockIdx.x];
					sp1[k+flag] = lindex - blockIdx.x*PSIZE; //src
#ifdef E1
					sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 1000;
#endif

//if(lindex - blockIdx.x*PSIZE < 0 || lindex - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", lindex);
					sp2[k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))]] = dst_v; //dst
					if(((k+flag)&7) == 0) {
						mapper[(k+flag)>>3] = k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))];
					}
				}                         }
                }
                // step 3 (large granularity)
                for(i=(threadIdx.x>>8);i<occ[SM_SIZE-1];i+=4) {
                        int lindex = buffer3[i];
                        int bf2 = csc_v[lindex+1] - csc_v[lindex];
                        bf2 = bf2 - (bf2&255);
                        for(j=(threadIdx.x&255);j<bf2;j+=256) {
                                int index_pnt = csc_e[csc_v[lindex]+j] / PSIZE;
				int k = atomicAdd(&occ[DFACTOR*index_pnt+(threadIdx.x&(DFACTOR-1))],1);
				int flag = pb1[blockIdx.x*np*DFACTOR+index_pnt*DFACTOR+(threadIdx.x&(DFACTOR-1))];
				int dst_v0 = csc_v[lindex]+j;
				int dst_v = csc_e[dst_v0] - index_pnt * PSIZE; //dst
//if(dst_v > PSIZE) printf("e : %d\n", dst_v);
				if(flag >= 0) { //sparse
				
					flag += es[blockIdx.x];
					sp1[k+flag] = lindex - blockIdx.x*PSIZE; //src
#ifdef E1
					sp1v[k+flag] = csc_ev[dst_v0]; //src
//sp1v[k+flag] = 10000;
#endif

//if(lindex - blockIdx.x*PSIZE < 0 || lindex - blockIdx.x*PSIZE >= PSIZE) printf("err_occ %d\n", lindex);
					sp2[k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))]] = dst_v; //dst
					if(((k+flag)&7) == 0) {
						mapper[(k+flag)>>3] = k+pb2[index_pnt*np*DFACTOR+blockIdx.x*DFACTOR+(threadIdx.x&(DFACTOR-1))];
					}
				} 
                        }
                }
                __syncthreads();
        }
}
*/



__global__ void preprocessing_step21(int DFACTOR, int np, int *pb2, int *dx, int *dy, short *sp2, int *dmap)
{
	int base_addr = pb2[dy[blockIdx.x]*DFACTOR*np+dx[blockIdx.x]*DFACTOR];	
	short i;
	if(threadIdx.x == 0)
		dmap[blockIdx.x] = base_addr;
	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
		sp2[base_addr+i] = i;
	}
}

__global__ void preprocessing_step3(int DFACTOR, int LOG_DFACTOR, int *dindex, short *fx, short *fy, int *docc)
{
	__shared__ int socc[PSIZE];

	int i;
//	int height;
	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
		socc[i] = 0;
	}
	__syncthreads();

//	if(threadIdx.x == 0) printf("error : %d %d\n", dindex[blockIdx.x<<LOG_DFACTOR], dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR]);
//	height = ((dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR] - dindex[blockIdx.x<<LOG_DFACTOR])>>LOG_BSIZE);
	for(i=dindex[blockIdx.x<<LOG_DFACTOR]+threadIdx.x;i<dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR];i+=blockDim.x) {
//if(i < 0) printf("errrr\n");
//if(fy[i] > PSIZE) printf("err : %d\n", fy[i]);
		if(fy[i] < SUNUSED) {
//if(fy[i] >= PSIZE || fy[i] < 0) printf("type1 err\n");
			atomicAdd(&socc[fy[i]], 1);
		}
	}
	__syncthreads();

	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
		if(socc[i] > 0) {
			socc[i]++;
		} 
//		if((threadIdx.x&1) == 0) socc[i] ++;
//		socc[i]++;
	}
	__syncthreads();
	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
		docc[blockIdx.x*PSIZE + i] = socc[i];
//		if(docc[blockIdx.x*PSIZE*2 + i] > PSIZE) printf("ee\n");
	}
}

/*
__global__ void preprocessing_step31(int *dindex, int *docc)
{
	int i, height;
	height = ((dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR] - dindex[blockIdx.x<<LOG_DFACTOR])>>LOG_BSIZE);

	for(i=threadIdx.x;i<PSIZE*2;i+=blockDim.x) {
//		docc[blockIdx.x*PSIZE*2 + i] += (docc[blockIdx.x*PSIZE*2 + i]/height);
		int t = docc[blockIdx.x*PSIZE*2+i];
		for(int t1 = t/height; t1 > 0; t1 /= height)
			t += t1;
		docc[blockIdx.x*PSIZE*2 + i ] = t;

	}
}*/

__global__ void
__launch_bounds__(BSIZE, 1) 
preprocessing_step4(int DFACTOR, int LOG_DFACTOR, int *dindex, short *fx, short *fy, const int * __restrict__ docc, short *dp1
#ifdef E1
, E1T *dp1v, E1T *fz
#endif
)
{
	__shared__ int socc[PSIZE];

	int i;
	int height, base_addr=dindex[blockIdx.x<<LOG_DFACTOR];
	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
		socc[i] = 0;
	}
	__syncthreads();

	height = ((dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR] - base_addr)>>LOG_BSIZE)-1;
//if((dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR] - base_addr) % BSIZE != 0) printf("not-divide\n");

	for(i=base_addr+threadIdx.x;i<dindex[(blockIdx.x<<LOG_DFACTOR)+DFACTOR];i+=blockDim.x) {
		if(fy[i] < SUNUSED) {
//if(fy[i] < 0 || fy[i] > PSIZE) printf("err : %d\n", fy[i]);
			int k, k2, kq, kr;
			while(1) {
				k = atomicAdd(&socc[fy[i]], 1);
				k2 = docc[blockIdx.x*PSIZE+fy[i]]+k;
//if(2*fy[i]+bias < PSIZE-1 && k2 > docc[blockIdx.x*PSIZE*2+2*fy[i]+bias+1]) printf("e00 : %d %d %d %d %d\n", height, fy[i], k2,docc[blockIdx.x*PSIZE*2+fy[i]*2+1+bias], docc[blockIdx.x*PSIZE*2+2*fy[i]+bias]); 
//if(2*fy[i]+bias == PSIZE-1 && k2 >= dindex[(blockIdx.x+1)<<LOG_DFACTOR] - dindex[blockIdx.x<<LOG_DFACTOR])
//printf("e01 : %d %d\n", k2, dindex[(blockIdx.x+1)<<LOG_DFACTOR] - dindex[blockIdx.x<<LOG_DFACTOR]);
				kq = k2/height;
				kr = k2 - kq*height;
//if(kq >= BSIZE || kr >= height) printf("e0 : (%d %d) %d %d\n", k2, height, kq, kr);
				if(k == 0) {
//if(BSIZE+base_addr+kr*BSIZE+kq >= dindex[(blockIdx.x+1)*DFACTOR]) printf("err (%d %d) %d %d\n", blockIdx.x, threadIdx.x, BSIZE+base_addr+kr*BSIZE+kq, dindex[(blockIdx.x+1)*DFACTOR]); 
//				if(k == 0 && ((bias == 0) ||
//				(bias == 1 && (docc[blockIdx.x*PSIZE*4+fy[i]*4+1] == docc[blockIdx.x*PSIZE*4+fy[i]*4])) ||
//				(bias == 2 && (docc[blockIdx.x*PSIZE*4+fy[i]*4+2] == docc[blockIdx.x*PSIZE*4+fy[i]*4])) ||
//				(bias == 3 && (docc[blockIdx.x*PSIZE*4+fy[i]*4+3] == docc[blockIdx.x*PSIZE*4+fy[i]*4]))
//				)) {
					dp1[base_addr+BSIZE+kr*BSIZE+kq] = -(fy[i]+1);
				} else {
					if(kr == 0) {
						dp1[base_addr+kq] = -(fy[i]+1);
					}
					dp1[base_addr+BSIZE+kr*BSIZE+kq] = fx[i];
#ifdef E1
					dp1v[base_addr+BSIZE+kr*BSIZE+kq] = fz[i];
#endif
					break;
				}
			}
		}
	}

}

__global__ void preprocessing_step41(int s3g, int LOG_DFACTOR, int *dcindex, int *dccindex)
{
	int index = blockIdx.x*blockDim.x+threadIdx.x;
	if(index <= s3g) {
		dccindex[index] = dcindex[index<<LOG_DFACTOR];
	}
}


void generate_MultiGraph(struct HYB *m, struct csc_package *inp)
{
	cudaError_t cuda_stat;

	float add_factor = 1.4;

        int nv = inp->nv, ne = inp->ne, np = inp->np;
        int upper_nv = inp->upper_nv, upper_np = inp->upper_np;
        int DFACTOR = inp->DFACTOR, LOG_DFACTOR = inp->LOG_DFACTOR;
        int *csc_v = inp->csc_v, *csc_e = inp->csc_e;
#ifdef E1
	int *csc_ev = inp->csc_ev;
#endif

	int upper_ne = ne + 8*np*np;
	int start_point;
	int *_csc_v, *_csc_e;
	int dcnt;

	short *_csc_occ;
	int *_group_occ;
	int *_es, *_ces, *_pb1, *_pb2, *_cpb2;
	int *_cey;
	int *_dcnt, *_dx, *_dy, *_dccindex, *_dcindex, *_dindex;
	short *_fx, *_fy, *_sp1, *_sp2;
	int *_mapper;
	int *_docc, *_dcocc;	
	short *_dp1;
	int *_gr, *_cgr;
	int *_itable, *_csc_size; //index, loc(size)
	int *_ncsc_v, *_ncsc_e;
#ifdef E1
	E1T *_csc_ev, *_ncsc_ev, *_sp1v, *_dp1v, *_fz;
#endif
	int *_dmap;

	int *tt; short *tt2;//will be remomved
	int *ncsc_v = (int *)malloc(sizeof(int)*(nv+1));
	int *ncsc_e = (int *)malloc(sizeof(int)*ne);
#ifdef E1
	E1T *ncsc_ev = (E1T *)malloc(sizeof(E1T)*ne);
#endif

	int *_buff;
	cudaMalloc((void **) &_buff, sizeof(int)*32*np);

	cuda_stat = cudaMalloc((void **) &_csc_v, sizeof(int)*(nv+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_csc_v %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_csc_e, sizeof(int)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_csc_e %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_csc_occ, sizeof(short)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_csc_occ %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_group_occ, sizeof(int)*GROUP_NUM);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_group_occ %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemcpy(_csc_v, csc_v, sizeof(int)*(nv+1), cudaMemcpyHostToDevice);
	cudaMemcpy(_csc_e, csc_e, sizeof(int)*ne, cudaMemcpyHostToDevice);
	cudaMemset(_csc_occ, 0, sizeof(short)*ne);
	cudaMemset(_group_occ, 0, sizeof(int)*GROUP_NUM);
	



//printf("gr : %d\n", upper_nv);
	cuda_stat = cudaMalloc((void **) &_gr, sizeof(int)*upper_nv*GROUP_NUM);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_gr %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_cgr, sizeof(int)*upper_nv*GROUP_NUM);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_cgr %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_gr, 0, sizeof(int)*upper_nv*GROUP_NUM);
	cudaMemset(_cgr, 0, sizeof(int)*upper_nv*GROUP_NUM);


	cuda_stat = cudaMalloc((void **) &_itable, sizeof(int)*(upper_nv+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_itable %s\n", cudaGetErrorString(cuda_stat)); }
	int *it = (int *)malloc(sizeof(int)*(upper_nv+1));
	for(int i=0;i<=upper_nv;i++) 
		it[i] = i;
	cudaMemcpy(_itable, it, sizeof(int)*(upper_nv+1), cudaMemcpyHostToDevice);

	cuda_stat = cudaMalloc((void **) &_csc_size, sizeof(int)*(nv+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_csc_size %s\n", cudaGetErrorString(cuda_stat)); }
//	cudaMemset(_itable, 0, sizeof(int)*(upper_nv+1));
	cudaMemset(_csc_size, 0, sizeof(int)*(nv+1));
	free(it);

#ifdef E1
	cuda_stat = cudaMalloc((void **) &_csc_ev, sizeof(E1T)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_csc_ev %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemcpy(_csc_ev, csc_ev, sizeof(E1T)*ne, cudaMemcpyHostToDevice); // detailed value may be needed
#endif



        CUDPPHandle theCudpp;
        cudppCreate(&theCudpp);
        CUDPPConfiguration config;
        config.op = CUDPP_ADD;
        config.datatype = CUDPP_INT;
        config.algorithm = CUDPP_SCAN;
        config.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan1 = 0;
        CUDPPResult res = cudppPlan(theCudpp, &scanplan1, config, DFACTOR*np*np+1, 1, 0);

        CUDPPHandle theCudpp2;
        cudppCreate(&theCudpp2);
        CUDPPConfiguration config2;
        config2.op = CUDPP_ADD;
        config2.datatype = CUDPP_INT;
        config2.algorithm = CUDPP_SCAN;
        config2.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan2 = 0;
        CUDPPResult res2 = cudppPlan(theCudpp2, &scanplan2, config2, DFACTOR*np*np+1, 1, 0);

/*
        CUDPPHandle theCudpp3;
        cudppCreate(&theCudpp3);
        CUDPPConfiguration config3;
        config3.op = CUDPP_ADD;
        config3.datatype = CUDPP_INT;
        config3.algorithm = CUDPP_SCAN;
        config3.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_INCLUSIVE;
        CUDPPHandle scanplan3 = 0;
        CUDPPResult res3 = cudppPlan(theCudpp3, &scanplan3, config3, upper_nv*GROUP_NUM, GROUP_NUM, upper_nv);
*/

	CUDPPHandle theCudpp33;
        cudppCreate(&theCudpp33);
        CUDPPConfiguration config33;
        config33.op = CUDPP_ADD;
        config33.datatype = CUDPP_INT;
        config33.algorithm = CUDPP_SCAN;
        config33.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan33 = 0;
        CUDPPResult res33 = cudppPlan(theCudpp33, &scanplan33, config33, upper_nv, 1, 0);



        CUDPPHandle theCudpp4;
        cudppCreate(&theCudpp4);
        CUDPPConfiguration config4;
        config4.op = CUDPP_ADD;
        config4.datatype = CUDPP_INT;
        config4.algorithm = CUDPP_SCAN;
        config4.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan4 = 0;
        CUDPPResult res4 = cudppPlan(theCudpp4, &scanplan4, config4, nv+1, 1, 0);

	CUDPPHandle theCudpp5;
        cudppCreate(&theCudpp5);
        CUDPPConfiguration config5;
        config5.op = CUDPP_ADD;
        config5.datatype = CUDPP_INT;
        config5.algorithm = CUDPP_SCAN;
        config5.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan5 = 0;

       
/* // 8*K
        CUDPPHandle theCudpp4;
        cudppCreate(&theCudpp4);
        CUDPPConfiguration config4;
        config4.op = CUDPP_ADD;
        config4.datatype = CUDPP_INT;
        config4.algorithm = CUDPP_SCAN;
        config4.options = CUDPP_OPTION_FORWARD | CUDPP_OPTION_EXCLUSIVE;
        CUDPPHandle scanplan4 = 0;
	int SS=24;
        CUDPPResult res4 = cudppPlan(theCudpp4, &scanplan4, config4, SS*SS, SS, SS);
	int *ain, *aout;
	tt = (int *)malloc(sizeof(int)*SS*SS);
	for(int i=0;i<SS*SS;i++) tt[i]=1;
	cuda_stat = cudaMalloc((void **) &ain, sizeof(int)*SS*SS);
	cuda_stat = cudaMalloc((void **) &aout,sizeof(int)*SS*SS);
	cudaMemcpy(ain, tt, sizeof(int)*SS*SS, cudaMemcpyHostToDevice);
	res4 = cudppMultiScan(scanplan4, aout, ain, SS, SS);
	cudaMemcpy(tt, aout, sizeof(int)*SS*SS, cudaMemcpyDeviceToHost);
	for(int i=0;i<SS*SS;i++) {
		fprintf(stdout, "%d ", tt[i]);
	} fprintf(stdout, "\n");
	exit(0);
*/

	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, 0);
	int mp = devProp.multiProcessorCount;
	m->mp = mp;
//	int s00g=ne/(128*SFACTOR), s00b=128;
	int s00g=mp*16, s00b=128;
	int s01g=mp*2, s01b=BSIZE; // strict split
//	int s02g=CEIL(nv, 128), s02b=128;
	int s02g=mp*16, s02b=128;
//	int s03g=CEIL(nv, 64), s03b=256, s03n=CEIL(nv,64)*64;
	int s03g=(nv+63)/64, s03b=256, s03n=CEIL(nv,64)*64;
	////int s03g=(nv+63)/64, s03b=128, s03n=CEIL(nv,64)*64;
	int s1g=np, s1b=BSIZE;
	int s11g=CEIL(np+1+127,128), s11b=128;
	int s2size = DFACTOR*np*np+1;
	int s3g, s3b=BSIZE;
	int totalsize_dense;
	int group_occ[GROUP_NUM];
	double trans_begin, trans_end;

//fprintf(stdout, "%d %d\n", nv, ne);

	double s_time = rtclock(); cudaDeviceSynchronize();

    if(inp->tra) {

	preprocessing_step00<<<s00g, s00b>>>(nv, _csc_v, _csc_occ);

//SHORT_PRINT(_csc_occ,500);

	preprocessing_step01<<<s01g ,s01b>>>(nv, upper_nv, _csc_occ, _group_occ, _gr);


	res33 = cudppScan(scanplan33, &_cgr[0], &_gr[0], upper_nv);
	res33 = cudppScan(scanplan33, &_cgr[upper_nv], &_gr[upper_nv], upper_nv);
	res33 = cudppScan(scanplan33, &_cgr[upper_nv*2], &_gr[upper_nv*2], upper_nv);
	res33 = cudppScan(scanplan33, &_cgr[upper_nv*3], &_gr[upper_nv*3], upper_nv);
	res33 = cudppScan(scanplan33, &_cgr[upper_nv*4], &_gr[upper_nv*4], upper_nv);
	res33 = cudppScan(scanplan33, &_cgr[upper_nv*5], &_gr[upper_nv*5], upper_nv);


//	res3 = cudppMultiScan(scanplan3, _cgr, _gr, upper_nv, GROUP_NUM);
	cudaMemcpyAsync(group_occ, _group_occ, sizeof(int)*GROUP_NUM, cudaMemcpyDeviceToHost);

//INT_PRINT(&_gr[upper_nv*5], 10000);
//INT_PRINT(&_cgr[upper_nv*5], 10000);
//return;

//INT_PRINT(_group_occ,6);


/*
INT_PRINT(&_gr[nv-100], 100);
INT_PRINT(&_gr[upper_nv+nv-100], 100);
INT_PRINT(&_gr[2*upper_nv+nv-100], 100);
INT_PRINT(&_gr[3*upper_nv+nv-100], 100);
INT_PRINT(&_gr[4*upper_nv+nv-100], 100);
INT_PRINT(&_gr[5*upper_nv+nv-100], 100);

INT_PRINT(&_cgr[nv-100], 100);
INT_PRINT(&_cgr[upper_nv]+nv-100, 100);
INT_PRINT(&_cgr[2*upper_nv+nv-100], 100);
INT_PRINT(&_cgr[3*upper_nv+nv-100], 100);
INT_PRINT(&_cgr[4*upper_nv+nv-100], 100);
INT_PRINT(&_cgr[5*upper_nv+nv-100], 100);
*/
//return;

//INT_PRINT(&_cgr[5*upper_nv], upper_nv);

	cudaFree(_gr);
//INT_PRINT(_group_occ,6);
#ifdef E1_NO
	cuda_stat = cudaMalloc((void **) &_ncsc_ev, sizeof(E1T)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_ncsc_ev %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_ncsc_ev, UIN, sizeof(E1T)*ne); // detailed value may be needed
#endif

        int cpu1 = group_occ[0];
        int cpu2 = cpu1+group_occ[1];
        int cpu3 = cpu2+group_occ[2];
        int cpu4 = cpu3+group_occ[3];
        int cpu5 = cpu4+group_occ[4];
        int cpu_threshold = cpu5;


	// _itable : for vertices reordering
	preprocessing_step02<<<s02g, s02b>>>(nv, upper_nv, _csc_v, _cgr, _itable, _csc_size, _csc_occ,
	cpu1, cpu2, cpu3, cpu4, cpu5);

	cudaFree(_cgr);
	cudaFree(_csc_occ);

//INT_PRINT(_group_occ,6);


	cuda_stat = cudaMalloc((void **) &_ncsc_v, sizeof(int)*(nv+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_ncsc_v %s\n", cudaGetErrorString(cuda_stat)); }
	int *_tmp_e;
        cuda_stat = cudaMalloc((void **) &_tmp_e, sizeof(int)*ne);
        if(cuda_stat != cudaSuccess) { fprintf(stderr, "_tmp_e %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_ncsc_e, sizeof(int)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_ncsc_e %s\n", cudaGetErrorString(cuda_stat)); }
#ifdef E1
	cuda_stat = cudaMalloc((void **) &_ncsc_ev, sizeof(E1T)*ne);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_ncsc_ev %s\n", cudaGetErrorString(cuda_stat)); }
#endif
//	cudaMemset(_ncsc_v, 0, sizeof(int)*(nv+1));
//	cudaMemset(_ncsc_e, 0, sizeof(int)*ne);


//INT_PRINT(_itable, 10000);

//INT_PRINT(_itable, 50);
//INT_PRINT(_csc_size, 50);
//return;

	res4 = cudppScan(scanplan4, _ncsc_v, _csc_size, 1+nv);

//INT_PRINT(&_csc_size[10-10],10000);

//	preprocessing_step03<<<s03g, s03b>>>(nv, s03n, _itable, _csc_v, _csc_e, _ncsc_v, _tmp_e, _ncsc_e
	preprocessing_step03r<<<s03g, s03b>>>(nv, s03n, _itable, _csc_v, _csc_e, _ncsc_v, _ncsc_e
//	,cpu1, cpu2, cpu3, cpu4, cpu5
#ifdef E1
	,_csc_ev, _ncsc_ev
#endif
	);

	cudaFree(_csc_v); cudaFree(_csc_e); cudaFree(_tmp_e);
#ifdef E1
	cudaFree(_csc_ev);
#endif

//INT_PRINT(_ncsc_e,1000);
//exit(0);	

    } else {
	cudaFree(_gr);
	cudaFree(_cgr);
	cudaFree(_csc_occ);
	_ncsc_v = _csc_v; _ncsc_e = _csc_e;
#ifdef E1
	_ncsc_ev = _csc_ev;
#endif
    }


	cuda_stat = cudaMalloc((void **) &_es, sizeof(int)*(np+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_es %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_ces, sizeof(int)*(np+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_ces %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_cey, sizeof(int)*(np+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_cey %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_pb1, sizeof(int)*np*np*DFACTOR);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_pb1 %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_pb2, sizeof(int)*(np*np*DFACTOR+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_pb2 %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_cpb2, sizeof(int)*(np*np*DFACTOR+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_cpb2 %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_es, 0, sizeof(int)*(np+1));
	cudaMemset(_ces, 0, sizeof(int)*(np+1));
	cudaMemset(_cey, 0, sizeof(int)*(np+1));
	cudaMemset(_pb1, 0, sizeof(int)*np*np*DFACTOR);
	cudaMemset(_pb2, 0, sizeof(int)*(np*np*DFACTOR+1));
	cudaMemset(_cpb2, 0, sizeof(int)*(np*np*DFACTOR+1));


	cuda_stat = cudaMalloc((void **) &_dcnt, sizeof(int));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dcnt %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dx, sizeof(int)*np*np);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dx %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dy, sizeof(int)*np*np);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dy %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dccindex, sizeof(int)*(np*np+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dccindex %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dcindex, sizeof(int)*(np*np*DFACTOR+1));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dcindex %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dindex, sizeof(int)*(np*np*DFACTOR+1));
	cudaMemset(_dcnt, 0, sizeof(int));
	cudaMemset(_dx, 0, sizeof(int)*np*np);
	cudaMemset(_dy, 0, sizeof(int)*np*np);
	cudaMemset(_dccindex, 0, sizeof(int)*(np*np+1)); //cummulative index
	cudaMemset(_dcindex, 0, sizeof(int)*(np*np*DFACTOR+1)); //cummulative index
	cudaMemset(_dindex, 0, sizeof(int)*(np*np*DFACTOR+1));



//fprintf(stderr, "ok2\n");

	if(nv <= 12000000) {
		preprocessing_step1<<<s1g, s1b>>>(DFACTOR, LOG_DFACTOR, nv, ne, np, upper_np, _ncsc_v, _ncsc_e, _es, _pb1, _pb2,
		_dcnt, _dindex, _dx, _dy);
	} else if(nv <= 17000000) {
		preprocessing_step1medium<<<s1g, s1b>>>(DFACTOR, LOG_DFACTOR, nv, ne, np, upper_np, _ncsc_v, _ncsc_e, _es, _pb1, _pb2,
		_dcnt, _dindex, _dx, _dy);
	} else {
		if(PSIZE <= 4096) {
			preprocessing_step1verybig<<<s1g, s1b>>>(_buff, DFACTOR, LOG_DFACTOR, nv, ne, np, upper_np, _ncsc_v, _ncsc_e, _es, _pb1, _pb2,
			_dcnt, _dindex, _dx, _dy);
		} else {
			preprocessing_step1big<<<s1g, s1b>>>(DFACTOR, LOG_DFACTOR, nv, ne, np, upper_np, _ncsc_v, _ncsc_e, _es, _pb1, _pb2,
			_dcnt, _dindex, _dx, _dy);
		}
	}

	DFACTOR = 1; LOG_DFACTOR=0;

//INT_PRINT(_pb1,np*np*DFACTOR);
//INT_PRINT(_pb1,50);
//INT_PRINT(_pb2,50);
//return;



//fprintf(stderr, "ok21\n");

	cudaMemcpyAsync(&dcnt, _dcnt, sizeof(int), cudaMemcpyDeviceToHost);	

//fprintf(stderr, "small1\n");
	res = cudppScan(scanplan1, _ces, _es, np+1);

//fprintf(stderr, "small2\n");
	res = cudppScan(scanplan1, _cpb2, _pb2 , s2size); // need to be added


//fprintf(stderr, "small3\n");
	res2 = cudppScan(scanplan2, _dcindex, _dindex, 1+dcnt);

//fprintf(stderr, "small4\n");

	cudaMemcpyAsync(&totalsize_dense, &_dcindex[dcnt], sizeof(int), cudaMemcpyDeviceToHost); 
//INT_PRINT(_dcindex, 100);
//totalsize_dense*=2;

	cudaFree(_dindex);cudaFree(_es); cudaFree(_pb2); 


//fprintf(stderr, "ok22\n");

	cuda_stat = cudaMalloc((void **) &_docc, (dcnt>>LOG_DFACTOR)*sizeof(int)*PSIZE*2);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_docc %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dcocc, (dcnt>>LOG_DFACTOR)*sizeof(int)*PSIZE*2);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dcocc %s\n", cudaGetErrorString(cuda_stat)); }

	preprocessing_step11<<<s11g, s11b>>>(np, DFACTOR, _cpb2, _cey);
//cudaMemset(_docc, 0, dcnt*sizeof(int)*PSIZE);
//INT_PRINT(_ces, 100);
//INT_PRINT(_pb1, 100);
//INT_PRINT(_pb2, 100);

//INT_PRINT(_ncsc_v,100);
//INT_PRINT(_ncsc_e,100);
	int ssize;
	cudaMemcpyAsync(&ssize, &_cey[np], sizeof(int), cudaMemcpyDeviceToHost);

//	ssize *=2;
//fprintf(stderr, "ssize : %d\n", ssize);

//fprintf(stderr, "ok3\n");

	cuda_stat = cudaMalloc((void **) &_fx, sizeof(short)*totalsize_dense);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_fx %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_fy, sizeof(short)*totalsize_dense);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_fy %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_fx, UIN, sizeof(short)*totalsize_dense);
	cudaMemset(_fy, UIN, sizeof(short)*totalsize_dense);

#ifdef E1
	cuda_stat = cudaMalloc((void **) &_fz, sizeof(E1T)*totalsize_dense);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_fz %s\n", cudaGetErrorString(cuda_stat)); }
#endif


	cuda_stat = cudaMalloc((void **) &_sp1, sizeof(short)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sp1 %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_sp1, UIN, sizeof(short)*ssize); // detailed value may be needed

	cuda_stat = cudaMalloc((void **) &_sp2, sizeof(short)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sp2 %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_mapper, sizeof(int)*CEIL(ssize,8));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_mapper %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_sp2, UIN, sizeof(short)*ssize);
	cudaMemset(_mapper, 0, sizeof(int)*CEIL(ssize,8));
#ifdef E1
	cuda_stat = cudaMalloc((void **) &_sp1v, sizeof(E1T)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sp1v %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_sp1v, UIN, sizeof(E1T)*ssize); // detailed value may be needed
#endif


//fprintf(stderr, "ok4\n");

    if(nv <= 12000000) {
	preprocessing_step2<<<s1g, s1b>>>(DFACTOR, nv, ne, upper_ne, np, upper_np, _ncsc_v, _ncsc_e, _pb1, _cpb2, _ces,
	dcnt, _dcindex, _dx, _dy, _sp1, _sp2, _fx, _fy, _mapper
#ifdef E1
	, _ncsc_ev, _sp1v, _fz
#endif
	);
    } else if(nv <= 17000000) {
	preprocessing_step2medium<<<s1g, s1b>>>(DFACTOR, nv, ne, upper_ne, np, upper_np, _ncsc_v, _ncsc_e, _pb1, _cpb2, _ces,
	dcnt, _dcindex, _dx, _dy, _sp1, _sp2, _fx, _fy, _mapper
#ifdef E1
	, _ncsc_ev, _sp1v, _fz
#endif
	);
    } else {
	if(PSIZE <= 4096) {
		preprocessing_step2verybig<<<s1g, s1b>>>(DFACTOR, nv, ne, upper_ne, np, upper_np, _ncsc_v, _ncsc_e, _pb1, _cpb2, _ces,
		dcnt, _dcindex, _dx, _dy, _sp1, _sp2, _fx, _fy, _mapper
#ifdef E1
		, _ncsc_ev, _sp1v, _fz
#endif
		);
	} else {
		preprocessing_step2big<<<s1g, s1b>>>(DFACTOR, nv, ne, upper_ne, np, upper_np, _ncsc_v, _ncsc_e, _pb1, _cpb2, _ces,
		dcnt, _dcindex, _dx, _dy, _sp1, _sp2, _fx, _fy, _mapper
#ifdef E1
		, _ncsc_ev, _sp1v, _fz
#endif
		);
	}

    }
	cudaDeviceSynchronize();
	trans_begin = rtclock(); // for very large graph
	cudaMemcpy(ncsc_v, _ncsc_v, sizeof(int)*(nv+1), cudaMemcpyDeviceToHost);
	cudaMemcpy(ncsc_e, _ncsc_e, sizeof(int)*ne, cudaMemcpyDeviceToHost);
#ifdef E1
	cudaMemcpy(ncsc_ev, _ncsc_ev, sizeof(E1T)*ne, cudaMemcpyDeviceToHost);
#endif
	cudaFree(_ncsc_v); cudaFree(_ncsc_e);
#ifdef E1
cudaFree(_ncsc_ev);
#endif
	trans_end = rtclock();	
	s3g = (dcnt >> LOG_DFACTOR);

	
//fprintf(stderr, "succbefore\n");

	if(dcnt > 0) {
		cuda_stat = cudaMalloc((void **) &_dmap, sizeof(int)*(s3g));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dmap %s\n", cudaGetErrorString(cuda_stat)); }

	cuda_stat = cudaMalloc((void **) &_dp1, totalsize_dense*sizeof(short));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dp1 %s\n", cudaGetErrorString(cuda_stat)); }
#ifdef E1
//	cuda_stat = cudaMalloc((void **) &_dp1, totalsize_dense*sizeof(short));
//	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dp1 %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_dp1v, totalsize_dense*sizeof(E1T));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_dp1v %s\n", cudaGetErrorString(cuda_stat)); }
#endif


//fprintf(stderr, "ok5\n");


		cudaMemset(_dp1, UIN, sizeof(short)*totalsize_dense);

		CUDPPResult res5 = cudppPlan(theCudpp5, &scanplan5, config5, s3g*PSIZE, s3g, PSIZE);

		preprocessing_step3<<<s3g, s3b>>>(DFACTOR, LOG_DFACTOR, _dcindex, _fx, _fy, _docc);		

//INT_PRINT(_dcindex, np*DFACTOR+1);
//TEMP_PRINT(_fx,57344);
//SHORT_PRINT(_fy,100000);
//INT_PRINT(_docc, 100);
//return;
//exit(0);
//fprintf(stderr, "ox1\n");	
		res5 = cudppMultiScan(scanplan5, _dcocc, _docc, PSIZE, s3g);
//fprintf(stderr, "ox2\n");	

//INT_PRINT(_dcocc, PSIZE*2);
//		preprocessing_step31<<<s3g, s3b>>>(_dcindex, _dcocc);

//INT_PRINT(_dcocc, PSIZE*2);
//return;

//SHORT_PRINT(_fx, 100);
//SHORT_PRINT(_fy, 100);
//INT_PRINT(_docc, 100);
//INT_PRINT(_dcocc, 100);
//return;

//fprintf(stderr, "ok6\n");


		preprocessing_step4<<<s3g, s3b>>>(DFACTOR, LOG_DFACTOR, _dcindex, _fx, _fy, _dcocc, _dp1
#ifdef E1
		, _dp1v, _fz
#endif
		);

		preprocessing_step41<<<(s3g+127)>>7, 128>>>(s3g, LOG_DFACTOR, _dcindex, _dccindex);

//INT_PRINT(_dcindex, 100);
//INT_PRINT(_dccindex, 100);
//return;

//INT_PRINT(&_dcocc[PSIZE],PSIZE);
//printf("\n");
//INT_PRINT(&_docc[PSIZE],PSIZE);
//return;




	

	}

cudaFree(_fx); cudaFree(_fy);
#ifdef E1
cudaFree(_fz);
#endif 

/*	cuda_stat = cudaMalloc((void **) &_sp2, sizeof(short)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sp2 %s\n", cudaGetErrorString(cuda_stat)); }
	cuda_stat = cudaMalloc((void **) &_mapper, sizeof(int)*CEIL(ssize,8));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_mapper %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_sp2, UIN, sizeof(short)*ssize);
	cudaMemset(_mapper, 0, sizeof(int)*CEIL(ssize,8));
#ifdef E1
	cuda_stat = cudaMalloc((void **) &_sp1v, sizeof(E1T)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sp1v %s\n", cudaGetErrorString(cuda_stat)); }
	cudaMemset(_sp1v, UIN, sizeof(E1T)*ssize); // detailed value may be needed
#endif
*/

if(dcnt > 0) {
	preprocessing_step21<<<s3g, BSIZE>>>(DFACTOR, np, _cpb2, _dx, _dy, _sp2, _dmap);
}
/*
	preprocessing_step22<<<s1g, s1b>>>(DFACTOR, nv, ne, upper_ne, np, upper_np, _ncsc_v, _ncsc_e, _pb1, _cpb2, _ces,
	dcnt, _dcindex, _dx, _dy, _sp1, _sp2, _fx, _fy, _mapper
#ifdef E1
	, _ncsc_ev, _sp1v, _fz
#endif
	);*/


	cudaDeviceSynchronize(); double e_time = rtclock();
//	fprintf(stdout, "preprocessing : %f ms,", (e_time - trans_end + trans_begin - s_time)*1000);
//exit(0);
//TEMP_PRINT(_ces, np+1);
////TEMP_PRINT(_pb1, DFACTOR*np*np);

	// disallocate non-used variables
	cudppDestroy(scanplan1); cudppDestroy(theCudpp);
	cudppDestroy(scanplan2); cudppDestroy(theCudpp2);
//	cudppDestroy(scanplan3); cudppDestroy(theCudpp3);
	cudppDestroy(scanplan33); cudppDestroy(theCudpp33);
	cudppDestroy(scanplan4); cudppDestroy(theCudpp4);
	cudaFree(_pb1); 
	cudaFree(_dcindex);
	cudaFree(_docc);
	cudaFree(_csc_size); 
	cudaFree(_cpb2);

	cudaMalloc((void **) &_ncsc_v, sizeof(int)*(nv+1));
	cudaMemcpy(_ncsc_v, ncsc_v, sizeof(int)*(nv+1), cudaMemcpyHostToDevice);
	cudaMalloc((void **) &_ncsc_e, sizeof(int)*(ne));
	cudaMemcpy(_ncsc_e, ncsc_e, sizeof(int)*ne, cudaMemcpyHostToDevice);
#ifdef E1
	cudaMalloc((void **) &_ncsc_ev, sizeof(E1T)*(ne));
	cudaMemcpy(_ncsc_ev, ncsc_ev, sizeof(E1T)*ne, cudaMemcpyHostToDevice);
#endif

	V1T *_i1;

	cuda_stat = cudaMalloc((void **) &_i1, sizeof(V1T)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_i1 %s\n", cudaGetErrorString(cuda_stat)); }

	m->nv = nv; m->ne = ne; m->np = np;
	m->upper_nv = upper_nv; m->upper_np = upper_np;
	m->upper_ne = (m->ne) + (m->np)*(m->np)*8;
//	m->start_point = inp->start_p; // will be changed

	m->_ncsc_v = _ncsc_v; m->_ncsc_e = _ncsc_e;
	m->_itable = _itable; m->_ces = _ces; m->_cey = _cey;
	m->_sp1 = _sp1; m->_sp2 = _sp2;
	m->_mapper = _mapper;
	m->dcnt = s3g;
	m->_dx = _dx; m->_dy = _dy; m->_dccindex = _dccindex; m->_dmap = _dmap;
	m->_dp1 = _dp1;
#ifdef E1
	m->_ncsc_ev = _ncsc_ev; m->_sp1v = _sp1v; m->_dp1v = _dp1v;
#endif

	m->_i1 = _i1; // dep

#ifdef V2
	V2T *_i2;

	cuda_stat = cudaMalloc((void **) &_i2, sizeof(V2T)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_i2 %s\n", cudaGetErrorString(cuda_stat)); }
	m->_i2 = _i2;
#endif
#ifdef V3
	V3T *_i3;

	cuda_stat = cudaMalloc((void **) &_i3, sizeof(V3T)*ssize);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_i3 %s\n", cudaGetErrorString(cuda_stat)); }
	m->_i3 = _i3;

#endif

        m->dsample_size = (m->np)/dsample_factor;
        m->dsample_dcnt = (m->dcnt)/dsample_factor;
        cudaStreamCreate(&(m->stream1));
        cudaStreamCreate(&(m->stream2));

	cuda_stat = cudaMalloc((void **) &(m->_sample_partial), sizeof(float)*(m->mp)*16*32);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_sample_partial %s\n", cudaGetErrorString(cuda_stat)); }
        cuda_stat = cudaMalloc((void **) &(m->_temp_front), sizeof(int)*SCATTER_FACTOR);
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_temp_front %s\n", cudaGetErrorString(cuda_stat)); }
        cudaMemset(m->_temp_front, 0, sizeof(int)*SCATTER_FACTOR);

	m->d_time = -1;
	cuda_stat = cudaMalloc((void **) &(m->_finished), sizeof(int));
	if(cuda_stat != cudaSuccess) { fprintf(stderr, "_finished %s\n", cudaGetErrorString(cuda_stat)); }

	m->fflag = 0;
	m->degree = inp->degree;

	m->ssize = ssize;

	free(ncsc_v); free(ncsc_e);
#ifdef E1
	free(ncsc_ev);
#endif

//exit(0);
}

/*
void refreshV(struct HYB *m)
{

#ifdef INT_T
	cudaMemset(_i1, UIN, sizeof(V1T)*ssize);
#endif
#ifdef FLOAT_T
	float *t0;
	t0 = (float *)malloc(sizeof(float)*ssize);
	for(int ik=0; ik<ssize; ik++) {
		t0[ik] = -10000;
//		t0[ik] = IUNUSED;
	}
	cudaMemcpy(_i1, t0, sizeof(float)*ssize, cudaMemcpyHostToDevice);
	free(t0);
#endif
#ifdef V2
	cudaMemset(_i2, 0, sizeof(V2T)*ssize);
#endif
#ifdef V3
	cudaMemset(_i3, 0, sizeof(V3T)*ssize);
#endif

}*/

