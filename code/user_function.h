#include "common.h"

void inline initValue(int source_vertex, GRAPHTYPE *m, struct vector_data *vd)
{
#if defined BFS || defined SSSP || defined BC
	VERTEX_FRONTIER

	cudaMemcpyAsync(&(m->start_point), &(m->_itable[source_vertex]), sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemset(&(vd->_vv[m->start_point]), 0, sizeof(V1T));	

	vd->h_fs[0] = 1; vd->htot_size = 1; vd->algo = SPARSE; vd->fhave = 1;
	int k = m->start_point;
	cudaMemcpyAsync(vd->_curr_f, &k, sizeof(int), cudaMemcpyHostToDevice);

#endif
#if defined BC
	int one_value = 1;

	cudaMalloc((void **) &(vd->_vv2), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
        cudaMalloc((void **) &(vd->_nvv2), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemset(vd->_vv2, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemset(vd->_nvv2, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemcpyAsync(&(vd->_vv2[m->start_point]), &one_value, sizeof(int), cudaMemcpyHostToDevice);

        vd->accum_frpoint = (int *)malloc(sizeof(int)*ITER_LIMIT);
//        vd->accum_frpoint[0] = 0;
        cudaMalloc((void **) &(vd->accum_fr), sizeof(int)*(m->nv));
        for(int i=0;i<ITER_LIMIT;i++) {
                vd->accum_frpoint[i] = -1;
        }
	vd->last_accum = 0;

	cudaMalloc((void **) &(vd->_vv3), sizeof(float)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemset(vd->_vv3, 0, sizeof(float)*(CEIL(m->nv,PSIZE))*PSIZE);
	

#endif
#if defined CC
	EDGE_FRONTIER

	vd->h_fs[0] = (m->ne); vd->htot_size = (m->ne); vd->algo = SPARSE; vd->fhave = 1;	
	int *tmp = (int *)malloc(sizeof(int)*MAX(m->ne,m->nv));
	for(int i=0;i<MAX(m->ne,m->nv);i++) 
		tmp[i] = i;
	
	cudaMemcpyAsync(vd->_vv, tmp, sizeof(V1T)*(m->nv), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(vd->_curr_f, tmp, sizeof(int)*(m->ne), cudaMemcpyHostToDevice);
	free(tmp);
	
#endif
#if defined PR_T || defined PR_D
	VERTEX_FRONTIER

	vd->algo = DENSE;
	int *tmp = (int *)malloc(sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	int *ttmp = (int *)malloc(sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);

	for(int i=0;i<m->nv;i++) 
		tmp[i] = i;
	cudaMemcpyAsync(vd->_curr_f, tmp, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
	vd->htot_size = m->nv; vd->h_fs[0]=m->nv; 

	for(int i=0;i<CEIL(m->nv,PSIZE)*PSIZE;i++)
		tmp[i] = 1;
	cudaMemcpyAsync(vd->_vfc, tmp, sizeof(int)*(CEIL((m->nv),PSIZE)*PSIZE), cudaMemcpyHostToDevice);


	float *tmp2 = (float *)malloc(sizeof(float)*(m->nv));
	for(int i=0;i<m->nv;i++)
		tmp2[i] = PR_INITIAL_VALUE;
	
//	cudaMemcpyAsync(vd->_vv2, m->degree, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
//	cudaMemcpyAsync(vd->_vv, tmp2, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);

	cudaMalloc((void **) &(vd->_vv2), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
        cudaMalloc((void **) &(vd->_nvv2), sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemset(vd->_vv, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
//	cudaMemset(vd->_vfc, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
//	cudaMemcpy(tmp, m->_itable, sizeof(int)*(m->nv), cudaMemcpyDeviceToHost);
//	for(int i=0;i<(m->nv);i++) {
//		ttmp[tmp[i]] = m->degree[i];
//	}


	cudaMemcpy(tmp, m->_ncsc_v, sizeof(int)*((m->nv)+1), cudaMemcpyDeviceToHost);
	for(int i=0;i<(m->nv);i++) {
//		if(ttmp[i] != (tmp[i+1]-tmp[i])) printf("err %d %d %d\n", i, ttmp[i], tmp[i+1]-tmp[i]);
		ttmp[i] = tmp[i+1]-tmp[i];
	}


	cudaMemcpy(vd->_vv2, ttmp, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
	cudaMemset(vd->_nvv, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	free(tmp); free(tmp2);
#endif
}

void inline initValue2(int source_vertex, GRAPHTYPE *m, struct vector_data *vd)
{
#if defined CC
	VERTEX_FRONTIER
#endif
}

void inline resetValue(int source_vertex, GRAPHTYPE *m, struct vector_data *vd)
{
#if defined BFS || defined SSSP || defined BC
	m->d_time = -1;
	m->fflag = 0;

//	cudaMemset(&(vd->_vv[m->start_point]), 0, sizeof(V1T));	
        cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);\

	vd->h_fs[0] = 1; vd->htot_size = 1; vd->algo = SPARSE; vd->fhave = 1;
	vd->h_fs[1]=0; vd->h_fs[2]=0; vd->h_fs[3]=0;
	int k = m->start_point;
	cudaMemcpyAsync(vd->_curr_f, &k, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemset(vd->_vv, UIN, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
        cudaMemset(vd->_nvv, UIN, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
        cudaMemset(vd->_vfc, -1, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);

	cudaMemset(m->_i1, UIN, sizeof(V1T)*(m->ssize));

	cudaMemset(&(vd->_vv[m->start_point]), 0, sizeof(V1T));	


#endif
#if defined BC
	int one_value = 1;
	cudaMemset(vd->_vv2, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemset(vd->_nvv2, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	cudaMemcpyAsync(&(vd->_vv2[m->start_point]), &one_value, sizeof(int), cudaMemcpyHostToDevice);
        for(int i=0;i<ITER_LIMIT;i++) {
                vd->accum_frpoint[i] = -1;
        }
	vd->last_accum = 0;
	cudaMemset(vd->_vv3, 0, sizeof(float)*(CEIL(m->nv,PSIZE))*PSIZE);

	cudaMemset(m->_i2, 0, sizeof(V2T)*(m->ssize));
	cudaMemset(m->_i3, 0, sizeof(V3T)*(m->ssize));
#endif
#if defined CC

	vd->h_fs[0] = (m->ne); vd->h_fs[1]=0; vd->h_fs[2]=0; vd->h_fs[3]=0;
	vd->htot_size = (m->ne); vd->algo = SPARSE; vd->fhave = 1;	
        cudaMemset(vd->_gl, 0, sizeof(int)*SCATTER_FACTOR);\

	int *tmp = (int *)malloc(sizeof(int)*MAX(m->ne,m->nv));
	for(int i=0;i<MAX(m->ne,m->nv);i++) 
		tmp[i] = i;
	
	cudaMemcpyAsync(vd->_vv, tmp, sizeof(V1T)*(m->nv), cudaMemcpyHostToDevice);
	cudaMemcpyAsync(vd->_curr_f, tmp, sizeof(int)*(m->ne), cudaMemcpyHostToDevice);
	free(tmp);
	
#endif
#if defined PR_T || defined PR_D

	vd->algo = DENSE;
	int *tmp = (int *)malloc(sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	int *ttmp = (int *)malloc(sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);

	for(int i=0;i<m->nv;i++) 
		tmp[i] = i;
	cudaMemcpyAsync(vd->_curr_f, tmp, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
	vd->htot_size = m->nv; vd->h_fs[0]=m->nv; vd->h_fs[1]=0; vd->h_fs[2]=0; vd->h_fs[3]=0;

	for(int i=0;i<CEIL(m->nv,PSIZE)*PSIZE;i++)
		tmp[i] = 1;
	cudaMemcpyAsync(vd->_vfc, tmp, sizeof(int)*(CEIL((m->nv),PSIZE)*PSIZE), cudaMemcpyHostToDevice);


	float *tmp2 = (float *)malloc(sizeof(float)*(m->nv));
	for(int i=0;i<m->nv;i++)
		tmp2[i] = PR_INITIAL_VALUE;
	
//	cudaMemcpyAsync(vd->_vv2, m->degree, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
//	cudaMemcpyAsync(vd->_vv, tmp2, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);

	cudaMemset(vd->_vv, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
//	cudaMemset(vd->_vfc, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
//	cudaMemcpy(tmp, m->_itable, sizeof(int)*(m->nv), cudaMemcpyDeviceToHost);
//	for(int i=0;i<(m->nv);i++) {
//		ttmp[tmp[i]] = m->degree[i];
//	}


        float *t0;
        t0 = (float *)malloc(sizeof(float)*(m->ssize));
        for(int ik=0; ik<(m->ssize); ik++) {
                t0[ik] = -10000;
        }
        cudaMemcpy(m->_i1, t0, sizeof(float)*(m->ssize), cudaMemcpyHostToDevice);
        free(t0);

	cudaMemcpy(tmp, m->_ncsc_v, sizeof(int)*((m->nv)+1), cudaMemcpyDeviceToHost);
	for(int i=0;i<(m->nv);i++) {
		ttmp[i] = tmp[i+1]-tmp[i];
	}

        cudaMemset(m->_i2, 0, sizeof(V2T)*(m->ssize));


	cudaMemcpy(vd->_vv2, ttmp, sizeof(int)*(m->nv), cudaMemcpyHostToDevice);
	cudaMemset(vd->_nvv, 0, sizeof(int)*(CEIL(m->nv,PSIZE))*PSIZE);
	free(tmp); free(tmp2); free(ttmp);
#endif
}


__device__ __forceinline__ void initialize(V1T *loc, V1T data
#ifdef V2
, V2T *loc2, V2T data2
#endif
)
{
#if defined BFS
	*loc = BFS_INF;
#endif
#if defined SSSP
	*loc = SSSP_INF;
#endif
#if defined BC
	*loc = BFS_INF;
	*loc2 = 0;
#endif
#if defined PR_T || defined PR_D
	*loc = 0;
#endif
}

__device__ __forceinline__ void get_partial_result(V1T *loc, V1T data
#ifdef V2
, V2T *loc2, V2T data2
#endif
#ifdef E1
, E1T weight
#endif
)
{
#if defined BFS 
	*loc = data+1;
#endif
#if defined SSSP
	*loc = data+weight;
#endif
#if defined BC
//if(data == 0) printf("%d\n", 0);
	*loc = data+1;
	*loc2 = data2;	
#endif


#if defined PR_T || defined PR_D
	*loc = data;
/*	if(data2 != 0) {
		*loc = data/(float)data2;
	} else {
		*loc = 0;
	}*/
#endif
}

__device__ __forceinline__ void accumulate(V1T *loc, V1T data, int attribute
#ifdef V2
, V2T *loc2, V2T data2
#endif
)
{
#if defined BFS 
	if(data <= attribute && data < *loc) {
//printf("((%d))\n", data);
		*loc = data;
	}
#endif
#if defined SSSP
	atomicMin(loc, data);
#endif
#if defined BC
	if(data <= attribute && data < *loc) {
		*loc = data;
	}
//	if(data < BFS_INF) *loc2 += data2;
	if((data == attribute) && ((*loc) == BFS_INF || (*loc) == attribute)) {
		atomicAdd(loc2, data2);
	}
#endif
#if defined PR_T || PR_D
	atomicAdd(loc, data);
#endif
}


__device__ __forceinline__ void accumulate_nonAtomics(V1T *loc, V1T data, int attribute
#ifdef V2
, V2T *loc2, V2T data2
#endif
)
{
#if defined BFS 
	if(data <= attribute && data < *loc) {
		*loc = data;
	}
#endif
#if defined SSSP
	if(data < *loc) {
		*loc = data;
	}	
#endif
#if defined BC
	if(data <= attribute && data < *loc) {
		*loc = data;
	}
	if((data == attribute) && ((*loc) == BFS_INF || (*loc) == attribute)) {
//		*loc = attribute;
		(*loc2) += data2;
	}


//	if(data < *loc) {
//		*loc = data;
//	}
//	if(data == attribute-1) {
//		(*loc2) += data2;
//	}	
#endif
#if defined PR_T || defined PR_D
	*loc += data;
#endif
}

__device__ __forceinline__ int update_condition(int *old_loc, int *new_loc, int data, int attribute, int *frontier
#ifdef V2
, V2T *old_loc2, V2T *new_loc2, V2T data2
#endif
)
{
#if defined BFS
	if(data < *old_loc) {
		*new_loc = data;
		return true;
	} else {
		return false;
	}
#endif
#if defined SSSP
/*	if(data < abs(*old_loc)) {
		*new_loc = data;
		return true;
	} else {
		*new_loc = -abs(*old_loc);
		return false;
	}*/
	if(data < (*old_loc)) {
		*new_loc = data;
		*frontier = attribute;
		return true;
	} else {
		return false;
	}
#endif
#if defined BC
//printf("-%d %d-\n", data, *old_loc);
//return true;
	if(data < (*old_loc)) {
		*old_loc = data;
		*old_loc2 = data2;
		return true;

	} else {
		return false;
	}
#endif
#if defined CC
	if((*new_loc) != (*new_loc2)) {
		(*new_loc) = (*new_loc2);
		return true;
	} 
	return false;
#endif
#if defined PR_D || defined PR_T
	return true;
#endif
}

__device__ __forceinline__ int fused_update_condition(V1T *old_loc, V1T *new_loc, V1T data, int attribute, int *frontier
#ifdef V2
, V2T *old_loc2, V2T *new_loc2, V2T data2
#endif
)
{
#if defined BFS
	if(data <= attribute && data < *old_loc) {
		*new_loc = data;
		return true;
	} else {
		return false;
	}
#endif
#if defined SSSP
	if(data < *old_loc) {
		atomicMin(new_loc, data);
		if(atomicMax(frontier, attribute) < attribute) {
			return true;			
		}
	}
	return false;
#endif
#if defined BC
	if((*old_loc) == BFS_INF || (*old_loc) == attribute) {
		atomicAdd(new_loc2, data2);
		if(atomicCAS(new_loc, BFS_INF, attribute) == BFS_INF) {
			return true;
		}
	}
	return false;
#endif
#if defined CC
	if(data != data2) {
		if((attribute&1) == 1) {
			if(data < data2) {
				*new_loc2 = data;
			} else {
				*new_loc = data2;
			}
		} else {
			if(data < data2) {
				*new_loc = data2;
			} else {
				*new_loc2 = data;
			}
		}
		return true;
	}
	return false;
#endif
#if defined PR_D
	atomicAdd(new_loc, data);
//	if(data2 != 0) atomicAdd(new_loc, data/(float)data2);
	return false;
/*	if(atomicAdd(*newloc, data) < INIT_VALUE) {
		return true;
	} else {
		return false;
	}*/
#endif
}




/*
__device__ inline int fused_update_condition(int *old_loc, int *new_loc, int data, int attribute, short *frontier)
{
#if defined BFS
	if(data <= attribute && data < *old_loc) {
		*new_loc = data;
		return true;
	} else {
		return false;
	}
#endif
#if defined SSSP
	if(data < *old_loc) {
		atomicMin(old_loc, data);
		if(atomicMaxShort(frontier, attribute) < attribute) {
			return true;			
		}
	}
	return false;
#endif
}*/


