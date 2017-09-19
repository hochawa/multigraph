//#include "common.h"
#include "user_function.h"

void release_MultiGraph(struct HYB *m) 
{
	cudaFree(m->_ces); cudaFree(m->_cey); 
        cudaFree(m->_dx); cudaFree(m->_dy); cudaFree(m->_dccindex);
        cudaFree(m->_sp1); cudaFree(m->_sp2); cudaFree(m->_mapper);
        cudaFree(m->_ncsc_v); cudaFree(m->_ncsc_e);
        cudaFree(m->_itable);
        cudaFree(m->_i1); 
	if((m->dcnt) > 0) {
		cudaFree(m->_dp1);	
	}

	//m->vv release
	
	cudaFree(m->_i1); //dep
}
