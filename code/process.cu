#include "user_function.h"
//#include "common.h"
#if defined BFS || defined CC || defined BC || defined SSSP

#define MODE_HYBRID
//#define TRACE

#if defined BFS || defined CC || defined BC 
	#define SAME_FRONTIER
#endif
#if defined SSSP
	#define POSITIVE_FRONTIER
#endif

/*
void initValue(struct HYB *m, int **vv) // int is dep
{
        cudaMalloc((void **) vv, sizeof(int)*CEIL(m->nv,PSIZE)*PSIZE);
        cudaMemset(*vv, UIN, sizeof(int)*CEIL(m->nv,PSIZE)*PSIZE);

	int *v = *vv;

        cudaMemcpyAsync(&(m->start_point), &(m->_itable[0]), sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemset(&v[m->start_point], 0, sizeof(int));
}

void initFrontier(struct HYB *m, int **curr_f, int **next_f, int **gl)
{
	cudaMalloc((void **) curr_f, sizeof(int)*(m->nv)*SCATTER_FACTOR);
	cudaMalloc((void **) next_f, sizeof(int)*(m->nv)*SCATTER_FACTOR);
	cudaMalloc((void **) gl, sizeof(int)*SCATTER_FACTOR);

	int *p_curr_f = *curr_f;
	int *p_gl = *gl;
	int h_fs[SCATTER_FACTOR]={1}, htot_size = 1;
	
	int k = m->start_point;

//int *p_next_f=*next_f;
//cudaMemset(p_curr_f, 0, sizeof(int)*(m->nv)*4);
//cudaMemset(p_next_f, 0, sizeof(int)*(m->nv)*4);

//	cudaMemcpyAsync(p_gl, h_fs, sizeof(int)*SCATTER_FACTOR, cudaMemcpyHostToDevice);
	cudaMemset(p_gl, 0, sizeof(int)*SCATTER_FACTOR);
	cudaMemcpyAsync(p_curr_f, &k, sizeof(int), cudaMemcpyHostToDevice);
}*/

__global__ void phase_sample(int *vv, float *sample_partial, int iter
#ifdef POSITIVE_FRONTIER
, int *vfc
#endif
)
{
	int index=blockIdx.x*blockDim.x+threadIdx.x;
#ifdef SAME_FRONTIER
	int val = vv[index];
#endif
#ifdef POSITIVE_FRONTIER
	int val = vfc[index];
#endif
	float fval;
	if(val == iter) fval = 1; else fval = 0;
        for(int offset = 16; offset > 0; offset = offset >> 1) {
                fval += __shfl_down(val, offset);
        }
        if(threadIdx.x == 0) sample_partial[blockIdx.x] = fval;
}

__global__ void phase_sample_reduction(int size, float *sample_partial)
{
	__shared__ float sval[32];

	float val;
	if(threadIdx.x < size) val = sample_partial[threadIdx.x];	
	else val = 0;
        for(int offset = 16; offset > 0; offset = offset >> 1) {
                val += __shfl_down(val, offset);
        }
	if((threadIdx.x&31) == 0) {
		sval[threadIdx.x>>5] = val;
	}
	__syncthreads();
	if(threadIdx.x < 32) {
		val = sval[threadIdx.x];
	        for(int offset = 16; offset > 0; offset = offset >> 1) {
                	val += __shfl_down(val, offset);
      		}
	}
	if(threadIdx.x == 0) {
		sample_partial[threadIdx.x] = val;
	}

}

__global__ void make_frontier(int nv, int iter, int *vv, int *f1, int *f2, int *f3, int *f4, int *pointer, int *vfc)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int warp_id = ((threadIdx.x>>5)&3);
	if(index < nv) {
#if defined SAME_FRONTIER
		if(vv[index] == iter) {
#endif
#if defined POSITIVE_FRONTIER
		if(vfc[index] == iter) {
#endif
			if(warp_id == 0) f1[atomicAggInc(&pointer[0])] = index;
			else if(warp_id == 1) f2[atomicAggInc(&pointer[1])] = index;
			else if(warp_id == 2) f3[atomicAggInc(&pointer[2])] = index;
			else f4[atomicAggInc(&pointer[3])] = index;
		}
	}
}

__global__ void pseudo_process_f1(int nv, int tf_size, int *csc_v, int *csc_e, int *gl, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, int *vv, int *nvv, int iter, int flag, int *vfc
#ifdef V2
, V2T *vv2, V2T *nvv2
#endif
#ifdef E1
, E1T *csc_ev
#endif
)
{
	__shared__ int buffer1[64], buffer2[64], buffer3[64], buffer4[64];
	__shared__ int buffer_p[2];

	int i, j;
	int base_addr = blockIdx.x*64 + (threadIdx.x>>2);
	int checker = (blockIdx.x & (ssample_factor-1));
	if(flag == 0 && checker > 0) return;
	else if(flag == 1 && checker == 0) return;

	int warp_id, index, index_size, bias;
	if(threadIdx.x < 2) {
		buffer_p[threadIdx.x] = 0;
	}
	__syncthreads();
		warp_id = ((threadIdx.x>>5)&3);
	if(base_addr < tf_size) {
		bias = 0;
		if(base_addr < nn1) index = f1[base_addr];
		else if(base_addr <nn1+nn2) index = f2[base_addr-nn1];
		else if(base_addr < nn1+nn2+nn3) index = f3[base_addr-nn1-nn2];
		else index = f4[base_addr-nn1-nn2-nn3];
			index_size = csc_v[index+1] - csc_v[index];
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
			int index_dst = csc_e[csc_v[index]+i];
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+i]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &nvv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}
		}
	}
	__syncthreads();
	for(i=(threadIdx.x>>5);i<(buffer_p[0]>>1);i+=8) {
		index = buffer1[i];
		int bf2 = buffer2[i];
		int bf22 = bf2 - (bf2&255);
		for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
			int index_dst = csc_e[csc_v[index]+j];
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &nvv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}

		}
	}
	for(i=0;i<buffer_p[1];i++) {
		index = buffer3[i];
		for(j=threadIdx.x;j<(buffer4[i]>>1);j+=blockDim.x) {
			int index_dst = csc_e[csc_v[index]+j];
			int tmp;
#ifdef V2
			int tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &nvv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}

		}
	}
}


// relaseValue can be added
__global__ void pseudo_phase1(int nv, int ne, int upper_ne, int np, short *sp1, int *vv, int *i1, int *ces, int *mapper
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *sp1v
#endif
)
{
        __shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif

	int sblock = blockIdx.x*dsample_factor+(dsample_factor>>1);
        int base_addr = blockIdx.x*PSIZE*dsample_factor;
        int i, index;
        short temp;

        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                sv[i] = vv[i+base_addr];
#ifdef V2
		sv2[i] = vv2[i+base_addr];
#endif
        }
        __syncthreads();

        for(i=ces[sblock]+threadIdx.x; i<ces[sblock]+((ces[sblock+1]-ces[sblock])>>2); i+=blockDim.x) {
                index = mapper[i>>3]+(threadIdx.x&7);
                temp = sp1[i];
                if(temp < PSIZE) {
			////i1[index] = sv[temp]+1;
			get_partial_result(&i1[index], sv[temp]
#ifdef V2
			,&i2[index], sv2[temp]
#endif
#ifdef E1
			, sp1v[i]
#endif
			);
		}
        }
}

// DFACTOR, LOG_DFACTOR will be removed
__global__ void pseudo_phase11(int nv, int np, int *vv, int *i1, int *dx, int *dy, int *dindex, short *dp1, int *dmap, int iter
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *dp1v
#endif
)
{
        __shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif
	int sblock = blockIdx.x*dsample_factor+(dsample_factor>>1);

        int i;
        int x_base = dx[sblock];//, y_base = dy[blockIdx.x];
//if(x_base < 0 || x_base >= np || y_base < 0 || y_base >= np) printf("err0\n");
        short curr_index = SUNUSED;
        int curr_v, temp_v;
#ifdef V2
	V2T curr_v2, temp_v2;
#endif
	initialize(&curr_v, 0
#ifdef V2
	, &curr_v2, 0
#endif
	);

        int dmap_index = dmap[sblock];


        for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
//if(x_base*PSIZE+i < 0 || x_base*PSIZE+i >= nv) printf("err1\n");
                sv[i] = vv[x_base*PSIZE+i];
#ifdef V2
		sv2[i] = vv2[x_base*PSIZE+i];
#endif
        }
        __syncthreads();

        for(i=dindex[sblock]+threadIdx.x;i<dindex[sblock]+((dindex[sblock+1]-dindex[sblock])>>2);i+=blockDim.x) {
                short edge_value = dp1[i];
                if(edge_value < 0) {
//if(edge_value < -1*PSIZE) printf("errxx\n");
                        if(curr_index != SUNUSED) {
				accumulate(&i1[dmap_index+curr_index], curr_v, iter
#ifdef V2
				, &i2[dmap_index+curr_index], curr_v2
#endif
				);
                        }
                        curr_index = -(edge_value+1);
			initialize(&curr_v, 0
#ifdef V2
			, &curr_v2, 0
#endif
			);
//                        curr_v = IUNUSED;
                } else if(edge_value != SUNUSED) {
			temp_v = sv[edge_value];
#ifdef V2
			temp_v2 = sv2[edge_value];
#endif
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, temp_v
#ifdef V2
			, &tmp2, temp_v2
#endif
#ifdef E1
			, dp1v[i]
#endif
			);
			accumulate_nonAtomics(&curr_v, tmp, iter
#ifdef V2
			, &curr_v2, tmp2
#endif
			);

                        //temp_v = sv[edge_value];
                        //if(temp_v + 1 < curr_v) curr_v = temp_v+1;
                }
        }
	if(curr_index != SUNUSED) {
		accumulate(&i1[dmap_index+curr_index], curr_v, iter
#ifdef V2
		, &i2[dmap_index+curr_index], curr_v2
#endif
		);
	}
        //if(curr_index != SUNUSED && curr_v < i1[dmap_index + curr_index])
        //        i1[dmap_index+curr_index] = curr_v;

}

//DFAFCTOR will be removed
__global__ void pseudo_phase2(int nv, int ne, int np, short *sp2, int *vv, int *dummy_vv, int *i1, int *ces, int *finished, int iter, int *vfc
#ifdef V2
, V2T *vv2, V2T *nvv2, V2T *i2 
#endif
)
{
        __shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif

	int sblock = blockIdx.x*dsample_factor+(dsample_factor>>1);
        int base_addr = sblock*PSIZE;
        int i;

        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
		initialize(&sv[i], IUNUSED
#ifdef V2
		, &sv2[i], 0
#endif
		);
                ////sv[i] = IUNUSED;
        }
        __syncthreads();

//if(cpb2[fac*blockIdx.x] < 0 || cpb2[fac*(blockIdx.x+1)] > 1648304) printf("errr %d %d %d\n", blockIdx.x, cpb2[fac*blockIdx.x], cpb2[fac*(blockIdx.x+1)]);

        for(i=ces[sblock]+threadIdx.x;i<ces[sblock]+((ces[sblock+1]-ces[sblock])>>2); i+=blockDim.x) {
                int ii=sp2[i];
                ////if(i1[i] != IUNUSED && ii < PSIZE && sv[ii] > i1[i])
                ////     sv[ii] = i1[i];
		if(i1[i] != IUNUSED && ii < PSIZE) {
#ifdef BFS
	                if(i1[i] <= iter && i1[i] < sv[ii]) {
	                        sv[ii] = i1[i];
	                }
#endif
#ifdef SSSP
	                atomicMin(&sv[ii], i1[i]);
#endif
#ifdef BC
	                if(i1[i] <= iter && i1[i] < sv[ii]) {
	                        sv[ii] = i1[i];
	                }
	                if((i1[i] == iter) && (sv[ii] == BFS_INF || sv[ii] == iter)) {
	                        atomicAdd(&sv2[ii], i2[i]);
	                }
#endif

/*			accumulate(&sv[ii], i1[i], iter
#ifdef V2
		, &sv2[ii], i2[i]
#endif
			);*/
		}
        }
        __syncthreads();


        int flag = false;
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
#ifdef BC
                if(sv[i] < vv[i+base_addr]) {
                        dummy_vv[i+base_addr] = sv[i];
//                        vv2[i+base_addr] = sv2[i];
                        flag = true;
                }
#endif
#ifdef SSSP
                if(sv[i] < vv[i+base_addr]) {
                        dummy_vv[i+base_addr] = sv[i];
                        flag = true;
                }
#endif
#ifdef BFS
                if(sv[i] < vv[i+base_addr]) {
                        dummy_vv[i+base_addr] = sv[i];
                        flag = true;
                }
#endif

/*		flag = update_condition(&vv[i+base_addr], &dummy_vv[i+base_addr], sv[i], iter, &vfc[i+base_addr]
#ifdef V2
		, &vv2[i+base_addr], &nvv2[i+base_addr], sv2[i]
#endif
		);*/


        }

        if(__syncthreads_or(flag)) {
                if(threadIdx.x == 0) {
                        (*finished) = 1;
                }
        }
}

__global__ void process_f1(int nv, int tf_size, int *csc_v, int *csc_e, int *gl, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, int *vv, int *nvv, int iter, int flag, int *vfc
#ifdef V2
, V2T *vv2, V2T *nvv2
#endif
#ifdef E1
, E1T *csc_ev
#endif
)
{
	__shared__ int buffer1[64], buffer2[64], buffer3[64], buffer4[64];
	__shared__ int buffer_p[2];

	int i, j;
	int base_addr = blockIdx.x*64 + (threadIdx.x>>2);
//	int checker = (blockIdx.x & (sample_factor-1));
//	if(flag == 0 && checker > 0) return;
//	else if(flag == 1 && checker == 0) return;

	int warp_id, index, index_size, bias;
	if(threadIdx.x < 2) {
		buffer_p[threadIdx.x] = 0;
	}
	__syncthreads();
		warp_id = ((threadIdx.x>>5)&3);
	if(base_addr < tf_size) {
		bias = 0;
		if(base_addr < nn1) index = f1[base_addr];
		else if(base_addr <nn1+nn2) index = f2[base_addr-nn1];
		else if(base_addr < nn1+nn2+nn3) index = f3[base_addr-nn1-nn2];
		else index = f4[base_addr-nn1-nn2-nn3];
			index_size = csc_v[index+1] - csc_v[index];
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
			int index_dst = csc_e[csc_v[index]+i];
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+i]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &vv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &vv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}

			//if(vv[index_dst] == IUNUSED) {
			//	nvv[index_dst] = iter;
			//	if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
			//	else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
			//	else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
			//	else n_f4[atomicAggInc(&gl[3])] = index_dst;
			//}
		}
	}
	__syncthreads();
	for(i=(threadIdx.x>>5);i<buffer_p[0];i+=8) {
		index = buffer1[i];
		int bf2 = buffer2[i];
		int bf22 = bf2 - (bf2&255);
		for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
			int index_dst = csc_e[csc_v[index]+j];
//if(index_dst < 0 || index_dst >= nv) printf("err\n");
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &vv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &vv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}

			//if(vv[index_dst] == IUNUSED) {
			//	nvv[index_dst] = iter;
			//	if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
			//	else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
			//	else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
			//	else n_f4[atomicAggInc(&gl[3])] = index_dst;
			//}	
		}
	}
	for(i=0;i<buffer_p[1];i++) {
		index = buffer3[i];
		for(j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
			int index_dst = csc_e[csc_v[index]+j];
//if(index_dst < 0 || index_dst >= nv) printf("err\n");
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, vv[index]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized
			if(fused_update_condition(&vv[index_dst], &vv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &vv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}

			//if(vv[index_dst] == IUNUSED) {
			//	nvv[index_dst] = iter;
			//	       if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
			//	       else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
			//	       else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
			//	       else n_f4[atomicAggInc(&gl[3])] = index_dst;
			//}
		}
	}
}


	// relaseValue can be added
__global__ void phase1(int nv, int ne, int upper_ne, int np, short *sp1, int *vv, int *i1, int *ces, int *mapper
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *sp1v
#endif
)
{
	__shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif

//	if(sampled == 1 && (blockIdx.x & (ssample-1)) == 0) return;

	int base_addr = blockIdx.x*PSIZE;
	int i, index;
	short temp;
	for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
		sv[i] = vv[i+base_addr];
#ifdef V2
		sv2[i] = vv2[i+base_addr];
#endif
	}
	__syncthreads();
//printf("(%d %d)\n", ces[blockIdx.x], ces[blockIdx.x+1]);
	for(i=ces[blockIdx.x]+threadIdx.x; i<ces[blockIdx.x+1]; i+=blockDim.x) {
//printf("(%d)\n", i);
		index = mapper[i>>3]+(threadIdx.x&7);
		temp = sp1[i];
		if(temp < PSIZE) {
//#ifdef BFS
//			i1[index] = sv[temp]+1;
//#endif
			////i1[index] = sv[temp]+1;
			get_partial_result(&i1[index], sv[temp]
#ifdef V2
			, &i2[index], sv2[temp]
#endif
#ifdef E1
			, sp1v[i]
#endif
			);

		}
	}
}
	// DFACTOR, LOG_DFACTOR will be removed
__global__ void phase11(int nv, int np, int *vv, int *i1, int *dx, int *dy, int *dindex, short *dp1, int *dmap, int iter
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *dp1v
#endif
)
{
	__shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif

	int i;
	int x_base = dx[blockIdx.x];//, y_base = dy[blockIdx.x];

//	if(sampled == 1 && (x_base & (ssample-1)) == 0) return;

//if(x_base < 0 || x_base >= np || y_base < 0 || y_base >= np) printf("err0\n");
	short curr_index = SUNUSED;
	int curr_v, temp_v;
#ifdef V2
	V2T curr_v2, temp_v2;
#endif
	initialize(&curr_v, 0
#ifdef V2
	, &curr_v2, 0
#endif
	);

	int dmap_index = dmap[blockIdx.x];
	for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
//if(x_base*PSIZE+i < 0 || x_base*PSIZE+i >= nv) printf("err1\n");
		sv[i] = vv[x_base*PSIZE+i];
#ifdef V2
		sv2[i] = vv2[x_base*PSIZE+i];
#endif
	}
	__syncthreads();

	for(i=dindex[blockIdx.x]+threadIdx.x;i<dindex[blockIdx.x+1];i+=blockDim.x) {
		short edge_value = dp1[i];
		if(edge_value < 0) {
//if(edge_value < -1*PSIZE) printf("errxx\n");
			if(curr_index != SUNUSED) {
/*
#ifdef BFS
				if(curr_v <= iter && curr_v < i1[dmap_index+curr_index]) {
					i1[dmap_index+curr_index] = curr_v;
				}
#endif
#ifdef SSSP
				atomicMin(&i1[dmap_index+curr_index], curr_v);
#endif
#ifdef BC
				if(curr_v <= iter && curr_v < i1[dmap_index+curr_index]) {
					i1[dmap_index+curr_index] = curr_v;
				}
				if((curr_v == iter && (i1[dmap_index+curr_index] == BFS_INF || i1[dmap_index+curr_index] == iter))) {
					atomicAdd(&i2[dmap_index+curr_index], curr_v2);
				}
#endif
*/

				accumulate(&i1[dmap_index+curr_index], curr_v, iter
#ifdef V2
				, &i2[dmap_index+curr_index], curr_v2
#endif
				);
			}
			curr_index = -(edge_value+1);
//if(curr_index < 0 || curr_index >= PSIZE) printf("erryy\n");
			initialize(&curr_v, 0
#ifdef V2
			, &curr_v2, 0
#endif
			);
//			curr_v = IUNUSED;
		} else if(edge_value != SUNUSED) {
			temp_v = sv[edge_value];
#ifdef V2
			temp_v2 = sv2[edge_value];
#endif
			int tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, temp_v
#ifdef V2
			, &tmp2, temp_v2
#endif
#ifdef E1
			, dp1v[i]
#endif
			);
			accumulate_nonAtomics(&curr_v, tmp, iter
#ifdef V2
			, &curr_v2, tmp2
#endif
			);
			//if(temp_v + 1 < curr_v) curr_v = temp_v+1;
		}
	}
	if(curr_index != SUNUSED) {
		accumulate(&i1[dmap_index+curr_index], curr_v, iter
#ifdef V2
		, &i2[dmap_index+curr_index], curr_v2
#endif
		);
	}
	
	//if(curr_index != SUNUSED && curr_v < i1[dmap_index + curr_index] && curr_v <= iter)
	//	i1[dmap_index+curr_index] = curr_v;
}

	//DFAFCTOR will be removed
__global__ void phase2(int nv, int ne, int np, short *sp2, int *vv, int *i1, int *ces, int *finished, int iter, int *vfc
#ifdef V2
, V2T *vv2, V2T *i2
#endif
)
{
	__shared__ int sv[PSIZE];
#ifdef V2
	__shared__ V2T sv2[PSIZE];
#endif

	int base_addr = blockIdx.x*PSIZE;
	int i;

        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                ////sv[i] = IUNUSED;
#ifdef BFS
//		sv[i] = BFS_INF;
#endif

		initialize(&sv[i], IUNUSED
#ifdef V2
		, &sv2[i], 0
#endif
		);

        }
        __syncthreads();

//if(cpb2[fac*blockIdx.x] < 0 || cpb2[fac*(blockIdx.x+1)] > 1648304) printf("errr %d %d %d\n", blockIdx.x, cpb2[fac*blockIdx.x], cpb2[fac*(blockIdx.x+1)]);

        for(i=ces[blockIdx.x]+threadIdx.x;i<ces[blockIdx.x+1]; i+=blockDim.x) {
                int ii=sp2[i];
/*
#ifdef BFS
		if(i1[i] < IUNUSED && ii < PSIZE) {
			sv[ii] = MIN(sv[ii], i1[i]);
		}
#endif
*/

		if(i1[i] != IUNUSED && ii < PSIZE) {
/*			accumulate(&sv[ii], i1[i], iter
#ifdef V2
			, &sv2[ii], i2[i]
#endif
			);
//printf("(%d %d)", i1[i], sv2[ii]);*/
#ifdef BFS
		if(i1[i] <= iter && i1[i] < sv[ii]) {
			sv[ii] = i1[i];
		}
#endif
#ifdef SSSP
		atomicMin(&sv[ii], i1[i]);
#endif
#ifdef BC
		if(i1[i] <= iter && i1[i] < sv[ii]) {
			sv[ii] = i1[i];
		}
		if((i1[i] == iter) && (sv[ii] == BFS_INF || sv[ii] == iter)) {
			atomicAdd(&sv2[ii], i2[i]);
		}
#endif
		}

        }
        __syncthreads();


        int flag = false;
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {

#ifdef BC
		if(sv[i] < vv[i+base_addr]) {
			vv[i+base_addr] = sv[i];
			vv2[i+base_addr] = sv2[i];
			flag = true;
		}
#endif
#ifdef SSSP
		if(sv[i] < vv[i+base_addr]) {
			vv[i+base_addr] = sv[i];
			flag = true;
#ifndef DENSE_MODE
			vfc[i+base_addr] = iter;
#endif
		}
#endif
#ifdef BFS
		if(sv[i] < vv[i+base_addr]) {
			vv[i+base_addr] = sv[i];
			flag = true;
		}
#endif

/*
		flag = update_condition(&vv[i+base_addr], &vv[i+base_addr], sv[i], iter, &vfc[i+base_addr]
#ifdef V2
		, &vv2[i+base_addr], &vv2[i+base_addr], sv2[i]
#endif
		);
*/
                ////if(vv[i+base_addr] > sv[i]) {
                ////        vv[i+base_addr] = sv[i];
                ////        flag = true;
                ////}
        }
//if(flag == true) printf("true %d\n", threadIdx.x);
        if(__syncthreads_or(flag)) {
                if(threadIdx.x == 0) {
                        (*finished) = 1;
                }
        }

}


//__global__ vv_kernel<<<(nv+255)>>8, 256>>>(m->nv, m->_vv, m->_finished); 


__global__ void vv_kernel(int nv, int htot_size, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, int *vv, int *nvv, int iter, int *gl)
{
	int base_addr = blockIdx.x*256 + threadIdx.x;
	int warp_id = ((threadIdx.x>>5)&3);
	int index;
	int dummy;
	
	if(base_addr < htot_size) {
		if(base_addr < nn1) index = f1[base_addr];
		else if(base_addr < nn1+nn2) index = f2[base_addr-nn1];
		else if(base_addr < nn1+nn2+nn3) index = f3[base_addr-nn1-nn2];
		else index = f4[base_addr-nn1-nn2-nn3];

//		if(index < 0 || index >= nv) printf("err1 : %d\n", index);
//		if(vv[index] < 0 || vv[index] >= nv) printf("err2 : %d\n", vv[index]);

		if(update_condition(&vv[index], &vv[index], dummy, dummy, &dummy
#ifdef V2
		,&vv[vv[index]], &vv[vv[index]], dummy
#endif
		)) {
	                if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index;
                        else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index;
                        else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index;
                        else n_f4[atomicAggInc(&gl[3])] = index;
		}
	}
}

__global__ void ee_kernel(int nv, int ne, int htot_size, int *v1, int *v2, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, int *vv, int *nvv, int iter, int *gl)
{
	int i,j, index;
	int base_addr = blockIdx.x*256 + threadIdx.x;
	int warp_id = ((threadIdx.x>>5)&3);
	int *dummy;

	if(base_addr < htot_size) {
		if(base_addr < nn1) index = f1[base_addr];
		else if(base_addr < nn1+nn2) index = f2[base_addr-nn1];
		else if(base_addr < nn1+nn2+nn3) index = f3[base_addr-nn1-nn2];
		else index = f4[base_addr-nn1-nn2-nn3];

		int src_value, dst_value;
		int index1=v1[index], index2=v2[index];
//		if(index1 < 0 || index1 >= nv || index2 < 0 || index2 >= nv) printf("err01 : %d %d\n", index1, index2);
		int val1 = vv[index1], val2 = vv[index2];
//		if(val1 < 0 || val1 >= nv || val2 < 0 || val2 >= nv) printf("err02 : %d %d\n", index1, index2);

		if(fused_update_condition(&vv[val1], &nvv[val1], val1, iter, dummy
#ifdef V2
		,&vv[val2], &nvv[val2], val2
#endif
		)) {
	                if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index;
                        else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index;
                        else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index;
                        else n_f4[atomicAggInc(&gl[3])] = index;
		}
	}
}

__global__ void cudaMemcpydd(int htot_size, int b_p, int *accum_fr, int nn1, int *f1, int nn2, int *f2, int nn3, int *f3, int nn4, int *f4)
{
        int base_addr = blockIdx.x*blockDim.x+threadIdx.x;

        if(base_addr < nn1) accum_fr[b_p+base_addr] = f1[base_addr]; // 4 atomics
        else if(base_addr < nn1+nn2) accum_fr[b_p+base_addr] = f2[base_addr-nn1];
        else if(base_addr < nn1+nn2+nn3) accum_fr[b_p+base_addr] = f3[base_addr-nn1-nn2];
        else if(base_addr < htot_size) accum_fr[b_p+base_addr] = f4[base_addr-nn1-nn2-nn3];
}


int MultiGraph_V_V(struct MULTI_SPARSE *m, struct vector_data *vd0, struct vector_data *vd, struct OUTPUT *out, int *g_iter)
{
	int iter=*g_iter;
	int ne = m->ne, nv = m->nv;
	int *_temp_f;

//	int finished = 0;
//	cudaMemcpyAync(m->_finished, &finished, sizeof(int), cudaMemcpyHostToDevice);
//	vv_kernel<<<(nv+255)>>8, 256>>>(m->nv, m->_vv, m->_finished); 


	cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);

//int *tt;
//INT_PRINT(vd->_curr_f, 20);

	vv_kernel<<<(vd->htot_size+255)>>8, 256>>>(m->nv, vd->htot_size, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
	vd->h_fs[1], &(vd->_curr_f[nv]), &(vd->_next_f[nv]), vd->h_fs[2], &(vd->_curr_f[nv*2]), &(vd->_next_f[nv*2]),
	vd->h_fs[3], &(vd->_curr_f[nv*3]), &(vd->_next_f[nv*3]), vd0->_vv, vd0->_vv, iter, vd->_gl);


//INT_PRINT(vd->_curr_f, 20);

	cudaMemcpyAsync(vd->h_fs, vd->_gl, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
        vd->htot_size = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);
	if ((vd->htot_size) == 0) { out->r = (vd->_vv); return false; }

//printf("%d %d %d %d %d\n", vd->h_fs[0], vd->h_fs[1], vd->h_fs[2], vd->h_fs[3], vd->htot_size);

	*g_iter = iter+1;
	_temp_f = (vd->_curr_f); (vd->_curr_f) = (vd->_next_f); (vd->_next_f) = _temp_f;



	out->r = (vd->_vv); //nvv (swap)
	return true;
}


int MultiGraph_E_E(struct MULTI_SPARSE *m, struct vector_data *vd, struct OUTPUT *out, int *g_iter)
{
	int iter=*g_iter;
	int ne = m->ne;
	int *_temp_f;	
	int *tt;

	cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);

//INT_PRINT(vd->_vv, 50);

	ee_kernel<<<(vd->htot_size+255)>>8, 256>>>(m->nv, m->ne, vd->htot_size, m->_p1, m->_p2, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
	vd->h_fs[1], &(vd->_curr_f[ne]), &(vd->_next_f[ne]), vd->h_fs[2], &(vd->_curr_f[ne+(ne>>1)]), &(vd->_next_f[ne+(ne>>1)]),
	vd->h_fs[3], &(vd->_curr_f[ne*2]), &(vd->_next_f[ne*2]), vd->_vv, vd->_vv, iter, vd->_gl);

	cudaMemcpyAsync(vd->h_fs, vd->_gl, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
        vd->htot_size = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);
	if ((vd->htot_size) == 0) { out->r = (vd->_vv); return false; }

//INT_PRINT(vd->_vv, 50);

	*g_iter = iter+1;
	_temp_f = (vd->_curr_f); (vd->_curr_f) = (vd->_next_f); (vd->_next_f) = _temp_f;

	out->r = (vd->_vv); //nvv (swap)
	return true;
}

int MultiGraph_V_E_V(struct HYB *m, struct csc_package *inp, struct vector_data *vd, struct OUTPUT *out, int *g_iter) // result : dep
{

	int iter=*g_iter;
	int finished=0;
	int palgo = vd->algo;

//	int *_vv=vd->_vv;
//	int *_curr_f=vd->_curr_f, *_next_f=vd->_next_f, *_gl=vd->_gl;
	int *_temp_f;
//	int *h_fs=vd->h_fs;

//	initValue(m, &_vv);

//	initFrontier(m, &(vd->_curr_f), &_next_f, &_gl);

//	cudaMalloc((void **) &_finished, sizeof(int));

	// sampling
//        cudaStream_t stream1=m->stream1, stream2=m->stream2;

#ifdef MODE_HYBRID

//	int *_dummy_vv=vd->_nvv;

	int d_threshold=IUNUSED, sampled = 0; // if added

	// add variables
	float sample_partial;
	//int *_temp_front;
	//cudaMalloc((void **) &_temp_front, sizeof(int)*SCATTER_FACTOR);
	//cudaMemset(_temp_front, 0, sizeof(int)*SCATTER_FACTOR);
#if defined SPARSE_MODE || defined DENSE_MODE
m->d_time = 100;
#endif

	if(m->d_time < 0) {
		cudaDeviceSynchronize();
		double sample_start = rtclock(); // start time

		pseudo_phase1<<<m->dsample_size,BSIZE,0,m->stream1>>>((m->nv), (m->ne), (m->upper_ne), (m->np), (m->_sp1), (vd->_vv), (m->_i1), (m->_ces), (m->_mapper)
#ifdef V2
		, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
		, (m->_sp1v)
#endif
		);
		pseudo_phase11<<<m->dsample_dcnt,BSIZE,0,m->stream2>>>((m->nv), (m->np), (vd->_vv), (m->_i1), (m->_dx), (m->_dy), (m->_dccindex), (m->_dp1), (m->_dmap), iter
#ifdef V2
		, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
		, (m->_dp1v)
#endif
		);
		pseudo_phase2<<<m->dsample_size,BSIZE>>>((m->nv), (m->ne), (m->np), (m->_sp2), (vd->_vv), vd->_nvv, (m->_i1), (m->_cey), m->_finished, iter, (vd->_vfc)
#ifdef V2
		, (vd->_vv2), (vd->_nvv2), (m->_i2)
#endif
		);

		cudaDeviceSynchronize();
		double sample_end = rtclock();
		(m->d_time) = (sample_end - sample_start) * dsample_factor * 4;
#ifdef TRACE
		fprintf(stdout, "sample : %f\n", 1000* ( m->d_time));
#endif
	}

#ifdef SPARSE_MODE
m->fflag = 2;
vd->algo = SPARSE;
#endif
#ifdef DENSE_MODE
m->fflag = 2;
vd->algo = DENSE;
#endif

//m->fflag = 2;
////m->fflag = 0;vd->algo = DENSE;

	if(m->fflag < 2) {
//	while(1) {
		if(vd->fhave == 0) { // don't have frontiter
//printf("should not happen\n");
			phase_sample<<<(m->mp)<<4,32>>>((vd->_vv), m->_sample_partial, iter-1
#ifdef POSITIVE_FRONTIER
			, (vd->_vfc)
#endif
			); //assuem dataset is not very small
			phase_sample_reduction<<<1,1024>>>((m->mp)<<4, m->_sample_partial);
			cudaMemcpyAsync(&sample_partial, m->_sample_partial, sizeof(float), cudaMemcpyDeviceToHost);	
			(vd->htot_size) = (int)(sample_partial * (m->nv) / ((m->mp)<<9));
			if((vd->htot_size) < ((m->nv)>>DENSE_THRESHOLD)) vd->algo = SPARSE;
//			else if((vd->htot_size) > d_threshold) algo = DENSE;
			else vd->algo = NOT_DETERMINED;
		} else {
			if((vd->htot_size) < ((m->nv)>>DENSE_THRESHOLD)) vd->algo = SPARSE;
////			else if((vd->htot_size) > d_threshold) algo = DENSE;
			else vd->algo = NOT_DETERMINED;
		}

//if(algo == NOT_DETERMINED) algo = DENSE;
//vd->algo = NOT_DETERMINED; // will be removed
//vd->algo = SPARSE;
//vd->algo = DENSE;
//vd->algo = DENSE;
		if(vd->algo == NOT_DETERMINED) {
			sampled = 1;
			// accumulate results
			if(vd->fhave == 0) {
				cudaMemset(m->_temp_front, 0, sizeof(int)*SCATTER_FACTOR);

				vd->fhave = 1;
				int grid_size = (((m->nv)+127)>>7); // sampling can be do(m->ne)
				make_frontier<<<grid_size, 128>>>((m->nv), iter-1, (vd->_vv), &(vd->_curr_f[0]), &(vd->_curr_f[(m->nv)]), &(vd->_curr_f[(m->nv)+((m->nv)>>1)]), &(vd->_curr_f[(m->nv)*2]), m->_temp_front, (vd->_vfc));
				cudaMemcpyAsync(vd->h_fs, m->_temp_front, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
				(vd->htot_size) = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);
			}

#if defined BFS || defined SSSP || defined CC || defined BC 
			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);
#endif
			cudaDeviceSynchronize();
			double sample_start2 = rtclock();
			pseudo_process_f1<<<((vd->htot_size)+255)>>6, 256>>>((m->nv), (vd->htot_size), (m->_ncsc_v), (m->_ncsc_e), vd->_gl, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
			vd->h_fs[1], &(vd->_curr_f[(m->nv)]), &(vd->_next_f[(m->nv)]), vd->h_fs[2], &(vd->_curr_f[(m->nv)+((m->nv)>>1)]), &(vd->_next_f[(m->nv)+((m->nv)>>1)]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*2]),
#if defined BFS || defined CC
			&(vd->_next_f[(m->nv)*2]), (vd->_vv), (vd->_vv), iter, 0, (vd->_vfc)
#endif
#if defined BC
			&(vd->_next_f[(m->nv)*2]), (vd->_vv), (vd->_nvv), iter, 0, (vd->_vfc)
#endif
#if defined SSSP
			&(vd->_next_f[(m->nv)*2]), (vd->_vv), (vd->_nvv), iter, 0, (vd->_vfc)
#endif
#ifdef V2
			, vd->_vv2, vd->_nvv2
#endif
#ifdef E1
			, (m->_ncsc_ev)
#endif
			);
#if defined BC 
			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);
#endif

			cudaDeviceSynchronize();
			double sample_end2 = rtclock();
			double s_time = (sample_end2 - sample_start2) * ssample_factor * 1;
			if((m->d_time) > s_time) vd->algo = SPARSE;
			else vd->algo = DENSE;

#ifdef TRACE
printf("%f %f ", s_time*1000, (m->d_time)*1000);
#endif
//sampled = 2;
		} else {
			sampled = 2;	
		} 

//printf("%d %d\n", iter, algo);
//if(d_threshold < IUNUSED) algo = DENSE;
//int kp = rand()%2;

//vd->algo = DENSE;
	}

//int kp;
//if((kp&1) == 0) vd->algo = SPARSE;
//else vd->algo = DENSE;
//vd->algo = SPARSE;
//m->fflag = 0;


		if(palgo != vd->algo) m->fflag++;

		if(vd->algo == DENSE) {
#ifdef TRACE
			cudaDeviceSynchronize();
			double local_start = rtclock();
#endif
			if(d_threshold == IUNUSED) d_threshold = MIN(d_threshold, (vd->htot_size));
//printf("((%d))\n", (vd->htot_size));
			vd->fhave = 0;
			finished = 0;
			cudaMemcpyAsync(m->_finished, &finished, sizeof(int), cudaMemcpyHostToDevice);
	
			phase1<<<(m->np),BSIZE,0,m->stream1>>>((m->nv), (m->ne), (m->upper_ne), (m->np), (m->_sp1), (vd->_vv), (m->_i1), (m->_ces), (m->_mapper)
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_sp1v)
#endif
			);
			phase11<<<(m->dcnt),BSIZE,0,m->stream2>>>((m->nv), (m->np), (vd->_vv), (m->_i1), (m->_dx), (m->_dy), (m->_dccindex), (m->_dp1), (m->_dmap), iter
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_dp1v)
#endif
			);

			phase2<<<(m->np),BSIZE>>>((m->nv), (m->ne), (m->np), (m->_sp2), (vd->_vv), (m->_i1), (m->_cey), m->_finished, iter, vd->_vfc
#ifdef V2
		, (vd->_vv2), (m->_i2)
#endif
			);
			cudaMemcpyAsync(&finished, m->_finished, sizeof(int), cudaMemcpyDeviceToHost);
//			if(finished == 0) break;
			if(finished == 0) {
				if(sampled == 2) { out->r = (vd->_vv); return false; }
				else {
					cudaMemcpyAsync(vd->h_fs, vd->_gl, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
					if((vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]) == 0) { out->r = (vd->_vv); return false; }
				}			
			}
			*g_iter = iter+1;
#ifdef TRACE
			cudaDeviceSynchronize();
			double local_end = rtclock();
			printf("0 %d %f\n", iter-1, (local_end-local_start)*1000);
//			printf("0 %d %f %d\n", iter-1, (local_end-local_start)*1000, d_threshold);
#endif
		} else {
			if(vd->fhave == 0) {
//printf("should not happen2\n");
				cudaMemset(m->_temp_front, 0, sizeof(int)*SCATTER_FACTOR);
				cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);
				vd->fhave = 1;
				int grid_size = (((m->nv)+127)>>7);
				make_frontier<<<grid_size, 128>>>((m->nv), iter-1, (vd->_vv), &(vd->_curr_f[0]), &(vd->_curr_f[(m->nv)]), &(vd->_curr_f[(m->nv)+((m->nv)>>1)]), &(vd->_curr_f[(m->nv)*2]), m->_temp_front, vd->_vfc);
				cudaMemcpyAsync(vd->h_fs, m->_temp_front, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
				(vd->htot_size) = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);
			}
#if defined BC
                        vd->accum_frpoint[iter] = (vd->last_accum)+(vd->htot_size);
                        cudaMemcpydd<<<((vd->htot_size+127)>>7), 128>>>(vd->htot_size, vd->last_accum, vd->accum_fr, vd->h_fs[0], &(vd->_curr_f[0]), vd->h_fs[1], &(vd->_curr_f[m->nv]), vd->h_fs[2],
			&(vd->_curr_f[(m->nv)+((m->nv)>>1)]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*2]));
                        vd->last_accum = vd->accum_frpoint[iter];
//			printf("(%d %d)\n", vd->last_accum, vd->accum_frpoint[iter]);
#endif

#ifdef TRACE
			cudaDeviceSynchronize();
			double local_start = rtclock();
#endif


			process_f1<<<((vd->htot_size)+255)>>6, 256>>>((m->nv), (vd->htot_size), (m->_ncsc_v), (m->_ncsc_e), vd->_gl, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
			vd->h_fs[1], &(vd->_curr_f[m->nv]), &(vd->_next_f[(m->nv)]), vd->h_fs[2], &(vd->_curr_f[(m->nv)+((m->nv)>>1)]), &(vd->_next_f[(m->nv)+((m->nv)>>1)]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*2]), &(vd->_next_f[(m->nv)*2]), (vd->_vv), (vd->_vv), iter, sampled, (vd->_vfc)
#ifdef V2
			, vd->_vv2, vd->_vv2
#endif
#ifdef E1
, (m->_ncsc_ev)
#endif
);
			
			cudaMemcpy(vd->h_fs, vd->_gl, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
			(vd->htot_size) = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);

			if((vd->htot_size) == 0) { out->r = (vd->_vv); return false; }

			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);			
			*g_iter = iter+1;
			_temp_f = (vd->_curr_f); (vd->_curr_f) = (vd->_next_f); (vd->_next_f) = _temp_f;
#ifdef TRACE
			cudaDeviceSynchronize();
			double local_end = rtclock();
			printf("1 %d %f\n", iter-1, (local_end-local_start)*1000);
#endif
		}		
//	}
//	cudaDeviceSynchronize();
//	double total_end = rtclock();
//	fprintf(stdout, "%d %f\n", iter, (total_end - sample_start)*1000);
#endif
	out->r = (vd->_vv);
#ifdef V2
	out->r2 = (vd->_vv2);
#endif
#ifdef V3
	out->r3 = (vd->_vv3);
#endif
	return true;
}
#endif

