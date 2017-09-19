//#include "common.h"
#include "user_function.h"

// need to be update (CSC)
// vv -> in HYB?
void verifyResults(struct csc_package *inp, GRAPHTYPE *m, struct OUTPUT *out, int source_vertex) // vv : dep
{
#ifdef VALIDATE

V1T *_vv = out->r;
int *tt;
//INT_PRINT(_vv,20);

	int *csc_v = inp->csc_v, *csc_e = inp->csc_e;
	int i;
#ifdef E1
	E1T *csc_ev = inp->csc_ev;
#endif
	int nv = m->nv;
        V1T *vv;
        vv = (V1T *)malloc(sizeof(V1T)*nv);
        cudaMemcpyAsync(vv, _vv, sizeof(V1T)*nv, cudaMemcpyDeviceToHost);

#ifndef CC
	int *_itable = m->_itable;
	int *itable;
        itable = (int *)malloc(sizeof(int)*nv);
        cudaMemcpyAsync(itable, _itable, sizeof(int)*nv, cudaMemcpyDeviceToHost);
#endif

        V1T *c_vv;
        c_vv = (V1T *)malloc(sizeof(V1T)*nv);

#if defined PR_D || defined PR_T
//	float *c_vv = (float *)malloc(sizeof(float)*nv);
	float *c_nvv = (float *)malloc(sizeof(float)*nv);
	memset(c_vv, 0, sizeof(float)*nv);
	
	int *vmark = (int *)malloc(sizeof(int)*nv);
	for(i=0;i<nv;i++)
		vmark[i]=1;

	int flag;



	while(1) {
		flag = false;
		memset(c_nvv, 0, sizeof(float)*nv);
		for(i=0;i<nv;i++) {
			int deg = (csc_v[i+1]-csc_v[i]);
			for(int j=csc_v[i]; j<csc_v[i+1]; j++) {
				int k = csc_e[j];
				if(deg > 0) 
					c_nvv[k] += c_vv[i]/(float)deg;
			}
		}

#ifdef PR_T
		for(i=0;i<nv;i++) {
	                float f1 = (1-PR_DAMPING_FACTOR) + c_nvv[i] * PR_DAMPING_FACTOR;
	    		float f2 = c_vv[i];
	                if(fabs(f1 - f2) > PR_TOLERANCE) {
	                        c_vv[i] = f1;
	                        flag = true;
	                } else {
				c_vv[i] = f2;
			}				
		}
#endif
#ifdef PR_D
		for(i=0;i<nv;i++) {
	                float f1 = (1-PR_DAMPING_FACTOR) + c_nvv[i] * PR_DAMPING_FACTOR;
	                float f2 = c_vv[i];
	                if(vmark[i] == 1 && fabs(f1 - f2) > PR_TOLERANCE*f2) {
	                                c_vv[i] = f1;
	                                flag = true;
	                } else {
	                                c_vv[i] = f1;
	                                vmark[i] = 0;
	                }
		}

#endif
		if(flag == false) break;
/*	
		for(i=0;i<100;i++) {
			fprintf(stdout, "%d ", c_vv[i]);
		} fprintf(stdout, "\n");*/

	}

	for(i=0;i<nv;i++) {
		int t = MAX(vv[itable[i]],c_vv[i]);
		if(t == 0) continue;
		if(abs(vv[itable[i]] - c_vv[i])/t > THR) {
			break;
//			printf("%d diff : %f %f\n", i, vv[itable[i]], c_vv[i]);
		}
	}
	if(i == nv) printf("validation : PASS\n");
	else printf("validation : FAIL\n");

#endif

#if defined BFS 
        int *c_q;
        int qhead=0, qtail=1;
        c_q = (int *)malloc(sizeof(int)*nv);
        memset(c_vv, UIN, sizeof(int)*nv);
        c_vv[source_vertex] = 0; c_q[0] = source_vertex; // 0 = start_point

        int c_iter=0;
        while(1) {
                if(qhead == qtail) break;
                c_iter++;
                i = c_q[qhead%nv];
                qhead++;
                for(int j=csc_v[i]; j<csc_v[i+1]; j++) {
                        int k = csc_e[j];
                        if(c_vv[k] == IUNUSED) {
                                c_vv[k] = c_vv[i]+1;
                                c_q[qtail%nv] = k;
                                qtail++;
                        }
                }
        }


#endif
#ifdef SSSP
        memset(c_vv, UIN, sizeof(int)*nv);
        c_vv[source_vertex] = 0; // 0 = start_point
	int conv;
	while(1) {
		conv = true;
		for(i=0; i<nv; i++) {
			for(int j=csc_v[i]; j<csc_v[i+1]; j++) {
				if(c_vv[i] + csc_ev[j] < c_vv[csc_e[j]]) {
					conv = false;
					c_vv[csc_e[j]] = c_vv[i] + csc_ev[j];
				}
			}
		}
		if(conv == true) break;
	}

#endif
#ifdef BC
        int *c_q;
        int qhead=0, qtail=1;
        c_q = (int *)malloc(sizeof(int)*nv*1.5);
        memset(c_vv, UIN, sizeof(int)*nv);
        c_vv[source_vertex] = 0; c_q[0] = source_vertex; // 0 = start_point

	int *c_vv2 = (int *)malloc(sizeof(int)*nv);
	memset(c_vv2, 0, sizeof(int)*nv);
	c_vv2[source_vertex] = 1;

	float *c_vv3 = (float *)malloc(sizeof(float)*nv);
	memset(c_vv3, 0, sizeof(int)*nv);


        int c_iter=0;
        while(1) {
                if(qhead == qtail) break;
                c_iter++;
                i = c_q[qhead];
                qhead++;
                for(int j=csc_v[i]; j<csc_v[i+1]; j++) {
                        int k = csc_e[j];
                        if(c_vv[k] == IUNUSED) {
                                c_vv[k] = c_vv[i]+1;
                                c_q[qtail] = k;
                                qtail++;
                        }
			if(c_vv[i] + 1 == c_vv[k]) {
				c_vv2[k] += c_vv2[i];
			}
                }
       }
	for(i=qtail-1; i>=0; i--) {
		for(int j=csc_v[c_q[i]]; j<csc_v[c_q[i]+1];j++) {
			int k = csc_e[j];
			if(c_vv[k] == c_vv[c_q[i]]-1) {
				c_vv3[k] += ((float)c_vv2[k]/c_vv2[c_q[i]])*(1.0f + c_vv3[c_q[i]]);
			}
		}	
	}

	


////fprintf(stdout, "***\n");
	int *_vv2 = out->r2;
	int *vv2 = (int *)malloc(sizeof(int)*nv);
	cudaMemcpyAsync(vv2, _vv2, sizeof(float)*nv, cudaMemcpyDeviceToHost);
	float *_vv3 = out->r3;
	float *vv3 = (float *)malloc(sizeof(float)*nv);
	cudaMemcpyAsync(vv3, _vv3, sizeof(float)*nv, cudaMemcpyDeviceToHost);

/*
//cpu code
	//vv1,vv2,vv3
	for(i=0;i<nv;i++) {
		int t = MAX(vv3[itable[i]],c_vv3[i]);
		if(t == 0) continue;
		if(abs(vv3[itable[i]] - c_vv3[i])/t > THR) {
			break;
//			printf("%d diff : %f %f\n", i, vv3[itable[i]], c_vv3[i]);
		}
	}

	if(i == nv) printf("PASS\n");
	else printf("FAIL\n");
*/
/*	FILE *fp2 = fopen("out2.txt", "w");
	for(i=0;i<nv;i++) {
		fprintf(fp2, "%d:\t%d\t%d\t%f\n", i, vv[itable[i]], vv2[itable[i]], vv3[itable[i]]);
	}
	fclose(fp2);*/
#endif
#ifdef CC
        int *c_q;
        c_q = (int *)malloc(sizeof(int)*nv);
	int kcnt=0;
	int *adjust_vv = (int *)malloc(sizeof(int)*nv);
	memset(adjust_vv, -1, sizeof(int)*nv);

int *temp_vv = (int *)malloc(sizeof(int)*nv);
for(i=0;i<nv;i++)
temp_vv[i] = vv[i];

	for(i=0;i<nv;i++) {
		if(adjust_vv[vv[i]] < 0) {
			adjust_vv[vv[i]] = kcnt;
			kcnt++;
		}
	}
	for(i=0;i<nv;i++) {
		vv[i] = adjust_vv[vv[i]];
	}
		

	for(i=0;i<nv;i++)
		c_vv[i] = i;


	while(1) {
		int pflag=0;
		for(i=0;i<nv;i++) {
			for(int j=csc_v[i];j<csc_v[i+1];j++) {
				if(c_vv[i] != c_vv[csc_e[j]]) {
					c_vv[i] = MIN(c_vv[i], c_vv[csc_e[j]]);
					c_vv[csc_e[j]] = MIN(c_vv[i], c_vv[csc_e[j]]);
					pflag=1;
				}
			}
		}
		if(pflag == 0) break;
	}
	
	memset(adjust_vv, -1, sizeof(int)*nv);

	kcnt= 0;
	for(i=0;i<nv;i++) {
		if(adjust_vv[c_vv[i]] < 0) {
			adjust_vv[c_vv[i]] = kcnt;
			kcnt++;
		}
	}
	for(i=0;i<nv;i++)
		c_vv[i] = adjust_vv[c_vv[i]];

//	memset(c_vv, -1, sizeof(int)*nv);
//	int kkcnt=0;
/*	while(1) {
		for(i=0;i<nv;i++) {
			if(c_vv[i] < 0) break;
		}
		if(i == nv) break;
	        int qhead=0, qtail=1;
	        c_vv[i] = kkcnt; c_q[0] = i;

	 	while(1) {
       	       		if(qhead == qtail) break;
       	         	i = c_q[qhead%nv];
       	         	qhead++;
                	for(int j=csc_v[i]; j<csc_v[i+1]; j++) {
                        	int k = csc_e[j];
                        	if(c_vv[k] < 0) {
                                	c_vv[k] = kkcnt;
                               		c_q[qtail%nv] = k;
                                	qtail++;
                        	}
                	}
        	}
		kkcnt++;
	}*/

	for(i=0;i<nv;i++) {
		if(vv[i] != c_vv[i]) break;
//		if(vv[i] != c_vv[i]) printf("%d %d %d %d\n", i, temp_vv[i], vv[i], c_vv[i]);
	}
	if(i == nv) fprintf(stdout, "validation : PASS\n");
	else fprintf(stdout, "validation : FAIL\n");

#endif

#if defined BFS || defined SSSP || defined BC
        for(i=0;i<nv;i++) {
            if(vv[itable[i]] != c_vv[i]) break;
//            if(vv[itable[i]] != c_vv[i]) printf("%d %d %d %d\n", i, itable[i], vv[itable[i]], c_vv[i]);
        }
        if(i == nv) fprintf(stdout, "validation : PASS\n");
        else fprintf(stdout, "validation : FAIL %d\n", i);
#endif



#ifdef PRINT_OUTPUT

	FILE *fp;
	fp = fopen("out.txt", "w");
	for(i=0;i<nv;i++) {
		fprintf(fp, "%d\t%f\n", i, vv[itable[i]]);
	} fclose(fp);

#endif
	

#else // VALIDATE
	printf("\n");
#endif // VALIDATE
}
