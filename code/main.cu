#ifndef _KERNEL_SUPPORT_
#define _KERNEL_SUPPORT_
#endif

#include "user_function.h"
#include <stdlib.h>
//#include "common.h"


struct csc_package inp;
struct HYB m;
struct MULTI_SPARSE m_sparse;
//struct MG status; // MG = multigraph
struct OUTPUT out;
struct vector_data vd;


int main(int argc, char **argv)
{

int *tt; short *tt2;
int u_loop = atoi(argv[3]);
double total_time=0.0f;
double total_ms=0.0f;

#if defined BFS || defined SSSP 

	generate_CSC(&inp, argv);

	generate_MultiGraph(&m, &inp);

	initValue(inp.start_p, &m, &vd);


    for(int loop=0;loop<u_loop;loop++) {
	int iter=1;
	resetValue(inp.start_p, &m, &vd);
	cudaDeviceSynchronize();

	double s1 = rtclock();
	while(MultiGraph_V_E_V(&m, &inp, &vd, &out, &iter)); 
	double s2 = rtclock();
	total_time += s2-s1;
    }
	total_ms = total_time*1000/(float)u_loop;	
	fprintf(stdout, "processing : %f ms,%f MTEPS,", total_ms, (float)inp.ne / (total_ms * 1000));

	verifyResults(&inp, &m, &out, inp.start_p);

	release_MultiGraph(&m);
#endif
#if defined BC

	generate_CSC(&inp, argv);

	generate_MultiGraph(&m, &inp);

	initValue(inp.start_p, &m, &vd);

//INT_PRINT(vd._vv2, m.nv);
//SHORT_PRINT(m._sp1, 10);
//SHORT_PRINT(m._sp2, 10);
//exit(0);

    for(int loop=0;loop<u_loop;loop++) {
 	int iter=1;
	resetValue(inp.start_p, &m, &vd);
	cudaDeviceSynchronize();
	double s1 = rtclock();
	while(MultiGraph_V_E_V(&m, &inp, &vd, &out, &iter)); 
	MultiGraph_V_E_V_pull(&m, &inp, &vd, &out, &iter); 
	double s2 = rtclock();
	total_time += s2-s1;
    }
	total_ms = total_time*1000/(float)u_loop;	
	fprintf(stdout, "processing : %f ms,%f MTEPS,", total_ms, (float)inp.ne * 2 / (total_ms * 1000));

//INT_PRINT(vd._vv, 100);

	verifyResults(&inp, &m, &out, inp.start_p);

//	verifyResults(&inp, &m, &out, 0);

//	release_MultiGraph(&m);

#endif
#if defined CC
	int iter=1, dummy;

	struct vector_data vvd;

	int *lst, *_lst;

	generate_CSC(&inp, argv);

	lst=(int *)malloc(sizeof(int)*(inp.nv));
	for(int i=0;i<inp.nv;i++) {
		lst[i] = i;
	}	
	cudaMalloc((void **) &_lst, sizeof(int)*(inp.nv));
	cudaMemcpyAsync(_lst, lst, sizeof(int)*(inp.nv), cudaMemcpyHostToDevice);

	generate_MultiGraphS(&m_sparse, &inp);

	initValue(0, &m_sparse, &vd);
	initValue2(0, &m_sparse, &vvd);

   for(int loop=0;loop<u_loop;loop++) {
	resetValue(inp.start_p, &m_sparse, &vd);
	cudaDeviceSynchronize();
	double s1 = rtclock();
	iter=1;
	while(1) {
		if(!MultiGraph_E_E(&m_sparse, &vd, &out, &iter)) break;
		vvd.h_fs[0] = (m_sparse.nv); vvd.h_fs[1] = 0; vvd.h_fs[2] = 0; vvd.h_fs[3] = 0;
		vvd.htot_size = vvd.h_fs[0]; 
		cudaMemcpyAsync(vvd._curr_f, _lst, sizeof(int)*(m_sparse.nv), cudaMemcpyDeviceToDevice);
		while(MultiGraph_V_V(&m_sparse, &vd, &vvd, &out, &dummy));
	}
	double s2 = rtclock();
	total_time += s2-s1;
   }
	total_ms = total_time*1000/(float)u_loop;	
	fprintf(stdout, "processing : %f ms,%f MTEPS,", total_ms, (float)inp.ne / (total_ms * 1000));
	verifyResults(&inp, &m_sparse, &out, 0);
#endif


#if defined PR_D || defined PR_T
	int iter;
	generate_CSC(&inp, argv);


//fprintf(stderr, "099990900\n");

	generate_MultiGraph(&m, &inp);

//fprintf(stderr, "099990900\n");

	initValue(0, &m, &vd);	


   for(int loop=0;loop<u_loop;loop++) {
	resetValue(inp.start_p, &m, &vd);
	cudaDeviceSynchronize();
	double s1 = rtclock();
	while(MultiGraph_V_E_V_NOTIDEM(&m, &inp, &vd, &out, &iter)) {
	}; 
	double s2 = rtclock();
	total_time += s2-s1;
    }

	total_ms = total_time*1000/(float)u_loop;	
	fprintf(stdout, "processing : %f ms,%f MTEPS,", total_ms, (float)inp.ne / (total_ms * 1000));

	verifyResults(&inp, &m, &out, 0);
#endif

}
