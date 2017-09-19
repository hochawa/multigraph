#include "user_function.h"
//#include "common.h"

#if defined PR_D || defined PR_T

#if defined BFS || defined CC || defined BC || defined PR_T || defined PR_D
	#define SAME_FRONTIER
#endif
#if defined SSSP
	#define POSITIVE_FRONTIER
#endif

__global__ void _phase_sample(int *vv, float *sample_partial, int iter
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

__global__ void _phase_sample_reduction(int size, float *sample_partial)
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

__global__ void _make_frontier(int nv, int iter, int *vv, int *f1, int *f2, int *f3, int *f4, int *pointer, int *vfc)
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


__global__ void _pseudo_phase2(int nv, int ne, int np, short *sp2, V1T *vv, V1T *i1, int *ces, int *finished, int iter, int *vfc
#ifdef V2
, V2T *vv2, V2T *i2
#endif
)
{
	__shared__ V1T sv[PSIZE];

	int base_addr = blockIdx.x*PSIZE;
	int i;
	int *dummy;

        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                ////sv[i] = IUNUSED;
#ifdef BFS
//		sv[i] = BFS_INF;
#endif
/*
		initialize(&sv[i], IUNUSED
#ifdef V2
		, dummy, 0
#endif
		);*/
		sv[i] = 0.0f;

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

		if(i1[i] > -0.01f && ii < PSIZE) {
			atomicAdd(&sv[ii], i1[i]);
/*
			accumulate(&sv[ii], i1[i], iter
#ifdef V2
			, dummy, *dummy
#endif
			);*/

//printf("(%d %d)", i1[i], sv2[ii]);
		}

        }
        __syncthreads();


        int flag = false;
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {

#ifdef PR_T
                float f1 = (1-PR_DAMPING_FACTOR) + sv[i] * PR_DAMPING_FACTOR;
                float f2 = vv[i+base_addr];
                if(fabs(f1 - f2) > PR_TOLERANCE) {
                        vv[i+base_addr] = f1;
                        flag = true;
                }
#endif
#ifdef PR_D
                float f1 = (1-PR_DAMPING_FACTOR) + sv[i] * PR_DAMPING_FACTOR;
                float f2 = vv[i+base_addr];
                if(vfc[i+base_addr] == 1 && fabs(f1 - f2) > PR_TOLERANCE*f2) {
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







__global__ void _process_f1(int nv, int tf_size, int *csc_v, int *csc_e, int *gl, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, V1T *vv, V1T *nvv, int iter, int flag, int *vfc
#ifdef V2
, V2T *vv2
#endif
#ifdef E1
, E1T *csc_ev
#endif
)
{
	__shared__ int buffer1[64], buffer2[64], buffer3[64], buffer4[64];
	float buffer5[64];
	__shared__ int buffer_p[2];

	int i, j;
	int index = blockIdx.x*64 + (threadIdx.x>>2);
	int base = blockIdx.x*64;
//	int checker = (blockIdx.x & (sample_factor-1));
//	if(flag == 0 && checker > 0) return;
//	else if(flag == 1 && checker == 0) return;

	int warp_id, index_size, bias;
	if(threadIdx.x < 2) {
		buffer_p[threadIdx.x] = 0;
	}
/*
	if(threadIdx.x < 64 && base+threadIdx.x < nv) {
		if(vv2[base+threadIdx.x] > 0) {
			buffer5[threadIdx.x] = vv[base+threadIdx.x] / (float)vv2[base+threadIdx.x];
		}
		else {
			buffer5[threadIdx.x] = 0;
		}
	}*/
	__syncthreads();
		warp_id = ((threadIdx.x>>5)&3);
	if(index < nv) {
		bias = 0;
			
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
		float delta;
		if(vv2[index] != 0) delta = vv[index]/(float)vv2[index];
		else delta = 0;
		for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
			int index_dst = csc_e[csc_v[index]+i];

/*			V1T tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, buffer5[index-base]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+i]
#endif
			); // bfs can be optimized
			int *dummy;*/
//			atomicAdd(&nvv[index_dst], buffer5[index-base]);
			if(vv2[index] != 0) atomicAdd(&nvv[index_dst], delta);


/*
			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, dummy, dummy, tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}*/

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

		float delta;
		if(vv2[index] != 0) delta = vv[index]/(float)vv2[index];
		else delta = 0;
		for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
			int index_dst = csc_e[csc_v[index]+j];
//if(index_dst < 0 || index_dst >= nv) printf("err\n");

/*			V1T tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, buffer5[index-base]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized
*/
			//atomicAdd(&nvv[index_dst], buffer5[index-base]);
			if(vv2[index] != 0) atomicAdd(&nvv[index_dst], delta);

/*			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &vv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}*/

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

		float delta;
		if(vv2[index] != 0) delta = vv[index]/(float)vv2[index];
		else delta = 0;


		for(j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
			int index_dst = csc_e[csc_v[index]+j];
//if(index_dst < 0 || index_dst >= nv) printf("err\n");
/*
			V1T tmp;
#ifdef V2
			V2T tmp2;
#endif
			get_partial_result(&tmp, buffer5[index-base]
#ifdef V2
			, &tmp2, vv2[index]
#endif
#ifdef E1
			, csc_ev[csc_v[index]+j]
#endif
			); // bfs can be optimized*/

			//atomicAdd(&nvv[index_dst], buffer5[index-base]);
			if(vv2[index] != 0) atomicAdd(&nvv[index_dst], delta);

/*			if(fused_update_condition(&vv[index_dst], &nvv[index_dst], tmp, iter, &vfc[index_dst]
#ifdef V2
			, &vv2[index_dst], &vv2[index_dst], tmp2
#endif
			)) {
				if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index_dst;
				else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index_dst;
				else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index_dst;
				else n_f4[atomicAggInc(&gl[3])] = index_dst;
			}*/

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


__global__ void process_f3(int nv, int tf_size, int *gl, int nn1, int *f1, int *n_f1, int nn2, int *f2, int *n_f2, int nn3, int *f3, int *n_f3, int nn4, int *f4, int *n_f4, V1T *vvc, V1T *n_vvc, int *v_front)
{
        int index = blockIdx.x*256 + (threadIdx.x); // blocksize = 128, 4thread=1vertex
        int warp_id;
if(index < nv) {

        warp_id = ((threadIdx.x>>5)&3);
        V1T ff1 = (1-PR_DAMPING_FACTOR) + n_vvc[index] * PR_DAMPING_FACTOR;
        V1T ff2 = vvc[index];
        if(v_front[index] == 1 && fabs(ff1 - ff2) > PR_TOLERANCE*ff2) {
                vvc[index] = ff1;
                if(warp_id == 0) n_f1[atomicAggInc(&gl[0])] = index;
                else if(warp_id == 1) n_f2[atomicAggInc(&gl[1])] = index;
                else if(warp_id == 2) n_f3[atomicAggInc(&gl[2])] = index;
                else n_f4[atomicAggInc(&gl[3])] = index;
        } else {
//      printf("no conv\n");
                vvc[index] = ff1;
                v_front[index] = 0;//reverse
        }
//	n_vvc[index] = 0;

}




}

	// relaseValue can be added
__global__ void _phase1(int nv, int ne, int upper_ne, int np, short *sp1, V1T *vv, V1T *i1, int *ces, int *mapper
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *sp1v
#endif
)
{
	__shared__ V1T sv[PSIZE];

//	if(sampled == 1 && (blockIdx.x & (ssample-1)) == 0) return;

	int base_addr = blockIdx.x*PSIZE;
	int i, index;
	short temp;
	for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
		sv[i] = vv[i+base_addr];
#ifdef V2
		if(vv2[base_addr+i] != 0) sv[i] = sv[i]/(float)vv2[base_addr+i];
		else sv[i] = 0;
#endif



	}
	__syncthreads();
//printf("(%d %d)\n", ces[blockIdx.x], ces[blockIdx.x+1]);
	for(i=ces[blockIdx.x]+threadIdx.x; i<ces[blockIdx.x+1]; i+=blockDim.x) {
//printf("(%d)\n", i);
		index = mapper[i>>3]+(threadIdx.x&7);
		temp = sp1[i];
		if(temp < PSIZE) {
//		int temp3 = sv2[temp];
			i1[index]=sv[temp];
//			if(temp3 != 0) i1[index]=sv[temp]/(float)temp3; else i1[index] = 0.0f;
//#ifdef BFS
//			i1[index] = sv[temp]+1;
//#endif
			////i1[index] = sv[temp]+1;
/*			get_partial_result(&i1[index], sv[temp]
#ifdef V2
			, &i2[index], sv2[temp]
#endif
#ifdef E1
			, sp1v[i]
#endif
			);*/

		}
	}
}
	// DFACTOR, LOG_DFACTOR will be removed
__global__ void _phase11(int nv, int np, V1T *vv, V1T *i1, int *dx, int *dy, int *dindex, short *dp1, int *dmap, int iter
#ifdef V2
, V2T *vv2, V2T *i2
#endif
#ifdef E1
, E1T *dp1v
#endif
)
{
	__shared__ V1T sv[PSIZE];

	int i;
	int x_base = dx[blockIdx.x];//, y_base = dy[blockIdx.x];

//	if(sampled == 1 && (x_base & (ssample-1)) == 0) return;

//if(x_base < 0 || x_base >= np || y_base < 0 || y_base >= np) printf("err0\n");
	short curr_index = SUNUSED;
	V1T curr_v, temp_v;
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
		if(vv2[x_base*PSIZE+i] != 0) sv[i] = sv[i]/(float)vv2[x_base*PSIZE+i];
		else sv[i] = 0;
#endif
//		sv2[i] = vv2[x_base*PSIZE+i];
		i1[dmap_index+i] = 0.0f;
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
				if(curr_v < iter && curr_v < i1[dmap_index+curr_index]) {
					i1[dmap_index+curr_index] = curr_v+1;
				}
				if((curr_v == iter-1 && (i1[dmap_index+curr_index] == BFS_INF || i1[dmap_index+curr_index] == iter))) {
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
			V1T tmp;
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
__global__ void _phase2(int nv, int ne, int np, short *sp2, V1T *vv, V1T *i1, int *ces, int *finished, int iter, int *vfc
#ifdef V2
, V2T *vv2, V2T *i2
#endif
)
{
	__shared__ V1T sv[PSIZE];

	int base_addr = blockIdx.x*PSIZE;
	int i;
	int *dummy;

        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                ////sv[i] = IUNUSED;
#ifdef BFS
//		sv[i] = BFS_INF;
#endif
/*
		initialize(&sv[i], IUNUSED
#ifdef V2
		, dummy, 0
#endif
		);*/
		sv[i] = 0.0f;

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

		if(i1[i] > -0.01f && ii < PSIZE) {
			atomicAdd(&sv[ii], i1[i]);
/*
			accumulate(&sv[ii], i1[i], iter
#ifdef V2
			, dummy, *dummy
#endif
			);*/

//printf("(%d %d)", i1[i], sv2[ii]);
		}

        }
        __syncthreads();


        int flag = false;
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {

#ifdef PR_T
                float f1 = (1-PR_DAMPING_FACTOR) + sv[i] * PR_DAMPING_FACTOR;
                float f2 = vv[i+base_addr];
                if(fabs(f1 - f2) > PR_TOLERANCE) {
                        vv[i+base_addr] = f1;
                        flag = true;
                }
#endif
#ifdef PR_D
                float f1 = (1-PR_DAMPING_FACTOR) + sv[i] * PR_DAMPING_FACTOR;
                float f2 = vv[i+base_addr];
                if(vfc[i+base_addr] == 1 && fabs(f1 - f2) > PR_TOLERANCE*f2) {
                                vv[i+base_addr] = f1;
                                flag = true;
                } else {
                                vv[i+base_addr] = f1;
                                vfc[i+base_addr] = 0;
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


int MultiGraph_V_E_V_NOTIDEM(struct HYB *m, struct csc_package *inp, struct vector_data *vd, struct OUTPUT *out, int *g_iter) // result : dep
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


//	int *_dummy_vv=vd->_nvv;

	int d_threshold=IUNUSED, sampled = 0; // if added

	// add variables
	float sample_partial;
	//int *_temp_front;
	//cudaMalloc((void **) &_temp_front, sizeof(int)*SCATTER_FACTOR);
	//cudaMemset(_temp_front, 0, sizeof(int)*SCATTER_FACTOR);
#if defined SPARSE_MODE || DENSE_MODE
m->d_time = 100;
#endif

	if(m->d_time < 0) {
		cudaDeviceSynchronize();
		double sample_start = rtclock(); // start time

			_phase1<<<(m->dsample_size),BSIZE,0,m->stream1>>>((m->nv), (m->ne), (m->upper_ne), (m->np), (m->_sp1), (vd->_vv), (m->_i1), (m->_ces), (m->_mapper)
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_sp1v)
#endif
			);
			_phase11<<<(m->dsample_dcnt),BSIZE,0,m->stream2>>>((m->nv), (m->np), (vd->_vv), (m->_i1), (m->_dx), (m->_dy), (m->_dccindex), (m->_dp1), (m->_dmap), iter
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_dp1v)
#endif
			);

			_pseudo_phase2<<<(m->dsample_size),BSIZE>>>((m->nv), (m->ne), (m->np), (m->_sp2), (vd->_vv), (m->_i1), (m->_cey), m->_finished, iter, vd->_vfc
#ifdef V2
		, (vd->_vv2), (m->_i2)
#endif
			);


		cudaDeviceSynchronize();
		double sample_end = rtclock();
		(m->d_time) = (sample_end - sample_start) * dsample_factor * 2;
#ifdef TRACE
		fprintf(stdout, "sample : %f\n", 1000* ( m->d_time));
#endif
	}

#if defined SPARSE_MODE
m->fflag = 2;
vd->algo = SPARSE;
#endif
#if defined DENSE_MODE
m->fflag = 2;
vd->algo = DENSE;
#endif		
	
		if(m->fflag < 1) {

			cudaDeviceSynchronize();
			double sample_start2 = rtclock();



			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);

			_process_f1<<<((vd->htot_size)+255)>>(6+2), 256>>>((m->nv), (vd->htot_size), (m->_ncsc_v), (m->_ncsc_e), vd->_gl, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
			vd->h_fs[1], &(vd->_curr_f[m->nv]), &(vd->_next_f[(m->nv)]), vd->h_fs[2], &(vd->_curr_f[(m->nv)*2]), &(vd->_next_f[(m->nv)*2]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*3]), &(vd->_next_f[(m->nv)*3]), (vd->_vv), (vd->_nvv), iter, sampled, (vd->_vfc)
#ifdef V2
			, vd->_vv2
#endif
#ifdef E1
, (m->_ncsc_ev)
#endif
);
			cudaDeviceSynchronize();
			double sample_end2 = rtclock();
			double s_time = (sample_end2 - sample_start2) * ssample_factor * 4;
			if((m->d_time) > s_time) vd->algo = SPARSE;
			else vd->algo = DENSE;

		}

		if(palgo == vd->algo) m->fflag++;

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
	
			_phase1<<<(m->np),BSIZE,0,m->stream1>>>((m->nv), (m->ne), (m->upper_ne), (m->np), (m->_sp1), (vd->_vv), (m->_i1), (m->_ces), (m->_mapper)
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_sp1v)
#endif
			);
			_phase11<<<(m->dcnt),BSIZE,0,m->stream2>>>((m->nv), (m->np), (vd->_vv), (m->_i1), (m->_dx), (m->_dy), (m->_dccindex), (m->_dp1), (m->_dmap), iter
#ifdef V2
			, (vd->_vv2), (m->_i2)
#endif
#ifdef E1
			, (m->_dp1v)
#endif
			);

			_phase2<<<(m->np),BSIZE>>>((m->nv), (m->ne), (m->np), (m->_sp2), (vd->_vv), (m->_i1), (m->_cey), m->_finished, iter, vd->_vfc
#ifdef V2
		, (vd->_vv2), (m->_i2)
#endif
			);
			cudaMemcpyAsync(&finished, m->_finished, sizeof(int), cudaMemcpyDeviceToHost);
//			if(finished == 0) break;
			if(finished == 0) {
				 out->r = (vd->_vv); return false; 
			}
			*g_iter = iter+1;
#ifdef TRACE
			cudaDeviceSynchronize();
			double local_end = rtclock();
			printf("0 %d %f\n", iter-1, (local_end-local_start)*1000);
//			printf("0 %d %f %d\n", iter-1, (local_end-local_start)*1000, d_threshold);
#endif
		} else {
#ifdef TRACE
			cudaDeviceSynchronize();
			double local_start = rtclock();
#endif

//printf("%d %d %d %d %d\n", vd->h_fs[0], vd->h_fs[1], vd->h_fs[2], vd->h_fs[3], vd->htot_size);
cudaMemset(vd->_nvv, 0, sizeof(float)*(m->nv));
cudaDeviceSynchronize();

			_process_f1<<<((m->nv)+255)>>6, 256>>>((m->nv), (vd->htot_size), (m->_ncsc_v), (m->_ncsc_e), vd->_gl, vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
			vd->h_fs[1], &(vd->_curr_f[m->nv]), &(vd->_next_f[(m->nv)]), vd->h_fs[2], &(vd->_curr_f[(m->nv)*2]), &(vd->_next_f[(m->nv)*2]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*3]), &(vd->_next_f[(m->nv)*3]), (vd->_vv), (vd->_nvv), iter, sampled, (vd->_vfc)
#ifdef V2
			, vd->_vv2
#endif
#ifdef E1
, (m->_ncsc_ev)
#endif
);

			cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);			

//printf("%d %d %d %d %d\n", vd->h_fs[0], vd->h_fs[1], vd->h_fs[2], vd->h_fs[3], vd->htot_size);

			process_f3<<<((m->nv)+255)>>8, 256>>>(m->nv, vd->htot_size, vd->_gl
			,vd->h_fs[0], &(vd->_curr_f[0]), &(vd->_next_f[0]),
			vd->h_fs[1], &(vd->_curr_f[m->nv]), &(vd->_next_f[(m->nv)]), vd->h_fs[2], &(vd->_curr_f[(m->nv)*2]), &(vd->_next_f[(m->nv)*2]), vd->h_fs[3], &(vd->_curr_f[(m->nv)*3]), &(vd->_next_f[(m->nv)*3]), (vd->_vv), (vd->_nvv), (vd->_vfc));

//printf("%d %d %d %d %d\n", vd->h_fs[0], vd->h_fs[1], vd->h_fs[2], vd->h_fs[3], vd->htot_size);
			
			cudaMemcpy(vd->h_fs, vd->_gl, sizeof(int)*SCATTER_FACTOR, cudaMemcpyDeviceToHost);
			(vd->htot_size) = (vd->h_fs[0])+(vd->h_fs[1])+(vd->h_fs[2])+(vd->h_fs[3]);

			if(vd->htot_size == 0) { (out->r) = (vd->_vv); return false; }


//if(iter == 50) return false;
//fprintf(stdout, "%d %d\n", iter, vd->htot_size);

			*g_iter = iter+1;
//			_temp_f = (vd->_curr_f); (vd->_curr_f) = (vd->_next_f); (vd->_next_f) = _temp_f;
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
