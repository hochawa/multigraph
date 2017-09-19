#include "user_function.h"
//#include "common.h"
#ifdef BC

__global__ void process_fbc(int nv, int iter, int tf_size, int *f1, int *vvc, int *v_count, float *v_dep, int *csr_v, int *csr_e)
{

        int i,j;
        int base_addr = blockIdx.x*64 + (threadIdx.x>>2); // blocksize = 128, 4thread=1vertex
        int warp_id, index, index_size, bias;

        __shared__ int buffer1[256], buffer2[256], buffer3[256], buffer4[256];
        __shared__ int buffer_p[2];
        __shared__ float s_buffer[8];
        if(threadIdx.x < 2) {
                buffer_p[threadIdx.x] = 0;
        }
        __syncthreads();

        warp_id = ((threadIdx.x>>5)&3);
if(base_addr < tf_size) {
        bias=0;
        index = f1[base_addr];

        index_size = csr_v[index+1] - csr_v[index];

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


        float v_center = (float)v_count[index];
        //float dep_s=v_dep[index]; int numpath = v_count[index];
        float v_dep_partial = 0.0;
        for(i=bias+(threadIdx.x&3); i<index_size; i+=4) {
                int index_dst = csr_e[csr_v[index]+i];
                if(vvc[index_dst] == iter+1) {
                        //atomicAdd(&v_dep[index_dst], ((float)v_count[index_dst]/numpath)*(1+dep_s));
                        //v_dep_partial += (((float)v_count[index_dst]/numpath)*(1+dep_s));
                        v_dep_partial += ((v_center/v_count[index_dst])*(1+v_dep[index_dst]));
                }
        }
        v_dep_partial += __shfl_down(v_dep_partial, 2);
        v_dep_partial += __shfl_down(v_dep_partial, 1);
        if((threadIdx.x&3) == 0) {
                v_dep[index] += v_dep_partial;
        }
}

        __syncthreads();

        for(i=(threadIdx.x>>5);i<buffer_p[0];i+=8) {
                index = buffer1[i];
                int bf2 = buffer2[i];
                int bf22 = bf2 - (bf2&255);

//              float dep_s=v_dep[index]; int numpath = v_count[index];
                float v_center = (float)v_count[index];
                float v_dep_partial = 0.0;
                for(j=bf22+(threadIdx.x&31);j<bf2;j+=32) {
                        int index_dst = csr_e[csr_v[index]+j];
                        if(vvc[index_dst] == iter+1) {
//                              atomicAdd(&v_dep[index_dst], ((float)v_count[index_dst]/numpath)*(1+dep_s));
                        //      v_dep_partial += (((float)v_count[index_dst]/numpath)*(1+dep_s));
                                v_dep_partial += ((v_center/v_count[index_dst])*(1+v_dep[index_dst]));
                        }
                }

                v_dep_partial += __shfl_down(v_dep_partial, 16);
                v_dep_partial += __shfl_down(v_dep_partial, 8);
                v_dep_partial += __shfl_down(v_dep_partial, 4);
                v_dep_partial += __shfl_down(v_dep_partial, 2);
                v_dep_partial += __shfl_down(v_dep_partial, 1);
                if((threadIdx.x&31) == 0) {
                        v_dep[index] += v_dep_partial;
                }

        }

        __syncthreads();

        for(i=0;i<buffer_p[1];i++) {
                index = buffer3[i];

                //float dep_s=v_dep[index]; int numpath = v_count[index];
                float v_center = (float)v_count[index];
                float v_dep_partial = 0.0;
                for(j=threadIdx.x;j<buffer4[i];j+=blockDim.x) {
                        int index_dst = csr_e[csr_v[index]+j];
                        if(vvc[index_dst] == iter+1) {
//                              atomicAdd(&v_dep[index_dst], ((float)v_count[index_dst]/numpath)*(1+dep_s));
//                              v_dep_partial += (((float)v_count[index_dst]/numpath)*(1+dep_s));
                                v_dep_partial += ((v_center/v_count[index_dst])*(1+v_dep[index_dst]));
                        }
                }
                v_dep_partial += __shfl_down(v_dep_partial, 16);
                v_dep_partial += __shfl_down(v_dep_partial, 8);
                v_dep_partial += __shfl_down(v_dep_partial, 4);
                v_dep_partial += __shfl_down(v_dep_partial, 2);
                v_dep_partial += __shfl_down(v_dep_partial, 1);
                if((threadIdx.x&31) == 0) {
                        s_buffer[threadIdx.x>>5] = v_dep_partial;
                }
                __syncthreads();
                if(threadIdx.x == 0) {
                        v_dep[index] += s_buffer[0]+s_buffer[1]+s_buffer[2]+s_buffer[3]+s_buffer[4]+s_buffer[5]+s_buffer[6]+s_buffer[7];
                }
                __syncthreads();
        }

}











__global__ void bc_phase1(int iter, int nv, int ne, int np, const short *v1, int *vv, int *v_count, float *v_dep, int *i1, int *i2, float *i3, const int *pb1, const int *__restrict__ mapper)
{
        int base_addr = blockIdx.x*PSIZE;
        int i, index; short temp2;
        __shared__ int sv[PSIZE];
        __shared__ int sv2[PSIZE];
        __shared__ float sv3[PSIZE];
        //initialize
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                sv[i] = vv[i+base_addr];
                sv2[i] = v_count[i+base_addr];
                sv3[i] = v_dep[i+base_addr];
        }
        __syncthreads();
//printf("th:%d bl:%d\n", threadIdx.x, blockIdx.x);
       //compute(scatter)
        for(i=pb1[blockIdx.x]+threadIdx.x; i<pb1[blockIdx.x+1]; i+=blockDim.x) {

                index = mapper[i>>3]+(threadIdx.x&7); // double-> /4
                temp2 = v1[i];
//printf("value : %d %d %d\n", temp2, PSIZE, index);
                if(temp2 < PSIZE) {
//printf("ok : %d\n", threadIdx.x);
                        //int pivot=sv[temp2];
//                      i1[index] = sv[temp2]; i2[index] = sv2[temp2]; i3[index] = sv3[temp2];
//                      i1[index] = sv[temp2]; // not needed
                        if(sv[temp2] == iter) i3[index] = (1.0f+sv3[temp2])/sv2[temp2];
                        else i3[index] = 0;
                }
        }
}

__global__ void bc_phase11(int iter, int dcnt, const int *dx, const int *dcum_size, const short *de, const int *dmap, const int *vv, const int *v_count, const float *v_dep, int *i1, int *i2, float *i3)
{
        __shared__ int sv[PSIZE];
        __shared__ int sv2[PSIZE];
        __shared__ float sv3[PSIZE];
//      __shared__ int dst_v[PSIZE];///
//int -> short (de)
        int i;
        int x_base = dx[blockIdx.x];
        short curr_index=SUNUSED;
	int curr_v=IUNUSED, temp_v;
	int base_index=dcum_size[blockIdx.x];
        int dmap_index=dmap[blockIdx.x];
        float curr_v2 = 0.0f;

        for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {
                sv[i] = vv[x_base*PSIZE+i];
                sv2[i] = v_count[x_base*PSIZE+i];
                sv3[i] = v_dep[x_base*PSIZE+i];
                //i3[dmap_index+i] = 0;
        }


        __syncthreads();

        for(i=base_index+threadIdx.x;i<dcum_size[blockIdx.x+1];i+=blockDim.x) {
                short edge_value = de[i];
                if(edge_value < 0) { // vertex
                        if(curr_index >= 0 && curr_index < PSIZE) {
                ////            if(curr_v < i1[dmap_index+curr_index]) {
                ////                    i1[dmap_index+curr_index] = curr_v;
                ////            }
                                if(curr_v2 > 0) {
                                        atomicAdd(&i3[dmap_index+curr_index], curr_v2);
                                }
                        }
                        curr_index = -(edge_value+1);
                        curr_v=BFS_INF;
                        curr_v2=0.0f;
                } else if(edge_value != SUNUSED) { // edge
                        temp_v = sv[edge_value];
                        if(temp_v < curr_v) curr_v = temp_v;
                        if(temp_v == iter) {
//printf(")%d\n", sv2[edge_value]);
                                curr_v2 += (1.0f + sv3[edge_value])/sv2[edge_value];
                        }
                }
        }
////    if(curr_index != UNUSED && curr_v < i1[dmap_index+curr_index])
////            i1[dmap_index+curr_index] = curr_v;
        if(curr_index >= 0 && curr_index < PSIZE && curr_v2 > 0)
                atomicAdd(&i3[dmap_index+curr_index], curr_v2);

/*      __syncthreads();///

        for(i=threadIdx.x;i<PSIZE;i+=blockDim.x) {///
                i1[dmap[blockIdx.x]+i] = dst_v[i];///
        }///
*/
}

__global__ void bc_phase2(int iter, int nv, int ne, int np, const short *v2, int *vv, int *v_count, float *v_dep, const int *i1, const int *i2, const float *i3, const int *pb2, int *finished)
{
        int base_addr = blockIdx.x*PSIZE;
        int i;
        __shared__ int sv[PSIZE];
        __shared__ int sv2[PSIZE];
        __shared__ float sv3[PSIZE];

        //initialize
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                sv[i] = vv[base_addr+i];
                sv2[i] = v_count[base_addr+i];
                sv3[i] = v_dep[base_addr+i];
        }
        __syncthreads();

        for(i=pb2[blockIdx.x]+threadIdx.x; i<pb2[blockIdx.x+1]; i+=blockDim.x) {
                int ii=v2[i];
                if(/*(i1[i] == iter) &&*/ (ii < PSIZE) && (sv[ii] == iter-1)) {
//printf("add\n");
//                      atomicAdd(&sv3[ii], ((float)sv2[ii]/i2[i])*(1+i3[i]));
                        atomicAdd(&sv3[ii], i3[i]*sv2[ii]);
                }
        }
        __syncthreads();

        //update
        for(i=threadIdx.x; i<PSIZE; i+=blockDim.x) {
                v_dep[base_addr+i] = sv3[i];
        }

}
#endif

#ifdef BC
int MultiGraph_V_E_V_pull(struct HYB *m, struct csc_package *inp, struct vector_data *vd, struct OUTPUT *out, int *g_iter) // result : dep
{
	int iter=*g_iter;

	int *accum_frpoint = vd->accum_frpoint, *accum_fr = vd->accum_fr;
	int nv = m->nv, ne = m->ne, np = m->np, dcnt = m->dcnt;
	int *vvc = vd->_vv, *v_countc = vd->_vv2, *ccsr_v = m->_ncsc_v, *ccsr_e = m->_ncsc_e;
	int *mapperc = m->_mapper;
	float *v_depc = vd->_vv3;

	short *v1c = m->_sp1, *v2c = m->_sp2;
	int *i1c = m->_i1, *i2c = m->_i2;
	float *i3c = m->_i3;

	int *d_dx = m->_dx, *d_dcum_size = m->_dccindex; short *d_de = m->_dp1;
	int *d_dmap = m->_dmap;

	int *pb1c = m->_ces;
	int *pb2c = m->_cey;
	int *finc;

	cudaStream_t stream1,stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);


/*
for(int i=0;i<iter;i++) {
	printf("%d\n", accum_frpoint[i]);
}*/

//return 0;
	accum_frpoint[0]=0;
	for(int i=1;i<=iter;i++) {
		if(accum_frpoint[i] == -1) accum_frpoint[i] = accum_frpoint[i-1];
	}

/*
printf("---\n");
for(int i=0;i<iter;i++) {
	printf("%d\n", accum_frpoint[i]);
}*/

	

        for(int i=iter-1; i>=0; i--) {
                if(accum_frpoint[i+1] > accum_frpoint[i]) {
                        process_fbc<<<((accum_frpoint[i+1]-accum_frpoint[i]+255)>>6),256>>>(nv, i, accum_frpoint[i+1]-accum_frpoint[i], &accum_fr[accum_frpoint[i]], vvc, v_countc, v_depc, ccsr_v, ccsr_e);
                } else {
                        bc_phase1<<<np,BSIZE,0,stream1>>>(i+1, nv, ne, np, v1c, vvc, v_countc, v_depc, i1c, i2c, i3c, pb1c, mapperc); // same as phase1, for efficiency
                        if(dcnt > 0) bc_phase11<<<dcnt,BSIZE,0,stream2>>>(i+1,dcnt,d_dx, d_dcum_size, d_de, d_dmap, vvc, v_countc, v_depc, i1c, i2c, i3c);
                        bc_phase2<<<np,BSIZE>>>(i+1, nv, ne, np, v2c, vvc, v_countc, v_depc, i1c, i2c, i3c, pb2c, finc);
                }
        }
	out->r = vd->_vv;
#ifdef V2
	out->r2 = vd->_vv2;
#endif
#ifdef V3
	out->r3 = vd->_vv3;
#endif
	return 0;
}

#endif
