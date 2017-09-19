//#include "common.h"
#include "user_function.h"

int compare1(const void *a, const void *b)
{
        if (((struct v_struct *)a)->src - ((struct v_struct *)b)->src > 0) return 1;
        if (((struct v_struct *)a)->src - ((struct v_struct *)b)->src < 0) return -1;
        return ((struct v_struct *)a)->dst - ((struct v_struct *)b)->dst;
}

void generate_CSC(struct csc_package *input, char **argv)
{
        FILE *fp;
        struct v_struct *temp_v;
        int *valid, *loc;
        char buf[300];
        int nflag, sflag;
        int dummy, pre_count=0, tmp_ne;
        int i;

	srand(time(NULL));

        // read "matrix market format"
        fp = fopen(argv[1], "r");
        fgets(buf, 300, fp);
//        if(strstr(buf, "symmetric") != NULL) sflag = 1; // symmetric
//        else sflag = 0;
//        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
//        else nflag = 1;

        if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
        else sflag = 0;
        if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
        else if(strstr(buf, "complex") != NULL) nflag = -1;
        else nflag = 1;



#ifdef SYM
sflag = 1;
#endif
#ifdef NSYM
sflag = 0;
#endif

        while(1) {
                pre_count++;
                fgets(buf, 300, fp);
                if(strstr(buf, "%") == NULL) break;
        }
        fclose(fp);

        fp = fopen(argv[1], "r");
	input->start_p = atoi(argv[2]);
        for(i=0;i<pre_count;i++)
                fgets(buf, 300, fp);

        fscanf(fp, "%d %d %d", &(input->nv), &dummy, &(input->ne));
        (input->ne) *= (sflag+1);
	(input->edges) = (input->ne);

        temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(input->ne));


	int *degree = (int *)malloc(sizeof(int)*(input->nv));
	memset(degree, 0, sizeof(int)*(input->nv));
	(input->tra) = 0;

        for(i=0;i<(input->ne);i++) {
                fscanf(fp, "%d %d", &temp_v[i].src, &temp_v[i].dst);
                temp_v[i].src--; temp_v[i].dst--;

                if(temp_v[i].src < 0 || temp_v[i].src >= (input->nv) || temp_v[i].dst < 0 || temp_v[i].dst >= (input->nv)) {
                        fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].src, temp_v[i].dst);
                        exit(0);
                }
                if(nflag == 0) temp_v[i].val = (rand()%64)+1;
                else if (nflag == 1) {
			float ftemp;
			fscanf(fp, " %f ", &ftemp); //need to be modified
			temp_v[i].val = (int)ftemp;
//printf("((%d))\n", temp_v[i].val);
		} else { // complex
			float ftemp1, ftemp2;
			fscanf(fp, " %f %f ", &ftemp1, &ftemp2);
			temp_v[i].val = (int)ftemp1;
		}
                if(sflag == 1) {
                        i++;
                        temp_v[i].dst = temp_v[i-1].src;
                        temp_v[i].src = temp_v[i-1].dst;
                        temp_v[i].val = temp_v[i-1].val;

			degree[temp_v[i].src]++;
                }
        }
        qsort(temp_v, input->ne, sizeof(struct v_struct), compare1);

        // remove duplicated edges & vertices not used
        loc = (int *)malloc(sizeof(int)*((input->ne)+1));
        valid = (int *)malloc(sizeof(int)*((input->nv)+1));

        memset(valid, 0, sizeof(int)*((input->nv)+1));
        memset(loc, 0, sizeof(int)*((input->ne)+1));
        loc[0]=1;
        for(i=1;i<(input->ne);i++) {
                valid[temp_v[i].src]=1; valid[temp_v[i].dst]=1;
                if(temp_v[i].src == temp_v[i-1].src && temp_v[i].dst == temp_v[i-1].dst)
                        loc[i] = 0;
                else loc[i] = 1;
#ifdef CC
		if(temp_v[i].src >= temp_v[i].dst) loc[i]=0;
#endif
        }
        for(i=1;i<=(input->nv);i++)
                valid[i] += valid[i-1];
        for(i=(input->nv); i>=1; i--)
                valid[i] = valid[i-1];
        valid[0] = 0;
        for(i=1;i<=(input->ne);i++)
                loc[i] += loc[i-1];
        for(i=(input->ne); i>=1; i--)
                loc[i] = loc[i-1];
        loc[0] = 0;

	for(i=0;i<input->ne;i++) {
		degree[temp_v[i].src]++;
	}

	input->start_p = valid[input->start_p];
	for(i=0;i<(input->nv);i++) {
		degree[valid[i]] = degree[i];
	}

        for(i=0;i<(input->ne);i++) {
                temp_v[loc[i]].src = valid[temp_v[i].src];
                temp_v[loc[i]].dst = valid[temp_v[i].dst];
                temp_v[loc[i]].val = temp_v[i].val;
        }

//for(i=0;i<(input->ne);i++) {
//printf("%d %d %d\n", temp_v[i].src, temp_v[i].dst, temp_v[i].val);
//}


//fprintf(stdout, "%d %d\n", input->nv, input->ne);
        (input->nv) = valid[input->nv]; (input->ne) = loc[input->ne];
//fprintf(stdout, "%d %d\n", input->nv, input->ne);
        (input->upper_nv) = CEIL(input->nv,32)*32;

        //convert COO to CSC
        (input->csc_v) = (int *)malloc(sizeof(int)*((input->nv)+1));
        (input->csc_e) = (int *)malloc(sizeof(int)*(input->ne));
#ifdef E1
	(input->csc_ev) = (E1T *)malloc(sizeof(E1T)*(input->ne));
#endif
        memset(input->csc_v, 0, sizeof(int)*((input->nv)+1));
        int temp_point=0;
        for(i=0;i<(input->ne);i++) {
                (input->csc_e[temp_point]) = temp_v[i].dst;
#ifdef E1
		(input->csc_ev[temp_point]) = temp_v[i].val;
#endif
                (input->csc_v[1+temp_v[i].src]) = temp_point+1;
                temp_point++;
        }
        for(i=1;i<(input->nv);i++) {
                if((input->csc_v[i]) == 0) (input->csc_v[i]) = (input->csc_v[i-1]);
		else if((input->csc_v[i]) - (input->csc_v[i-1]) >= 128) input->tra = 1;
	}
        (input->csc_v[(input->nv)]) = (input->ne);

        (input->np) = CEIL(input->nv, PSIZE);

//for(i=0;i<input->ne;i++) {
//printf("%d ", (input->csc_ev[i]));
//} printf("\n");

//        int lower_SM = ((SM_SIZE-2)/np_base)*np_base;

//        for((input->DFACTOR)=1,(input->LOG_DFACTOR)=0;(input->DFACTOR)*(input->np)*2 <= lower_SM/2; (input->DFACTOR) *= 2, (input->LOG_DFACTOR)++);

	(input->DFACTOR)=1; (input->LOG_DFACTOR)=0;

//	int nnp = CEIL(input->np, 8);
	int nnp = input->np;

	while(1) {
//		int occ = (input->DFACTOR)*(input->np);
		int occ = (input->DFACTOR)*nnp;
		int upper_occ = CEIL(occ, 512)*512;
		upper_occ += (upper_occ >> 9);
//		int upper_occ = occ + 9;
//		upper_occ = CEIL(occ, 512)*512;
 	    if((input->nv) <= 12000000) {
		if(upper_occ + 2 > SM_SIZE) break;
	    } else {
		if((input->nv) > 17000000 && PSIZE <= 4096) {
			if(upper_occ > SM_SIZE00 + 12) break;
		} else {
			if(upper_occ + 2 > SM_SIZE0) break;
		}
	    }
		(input->DFACTOR) *= 2;
		(input->LOG_DFACTOR)++;
//if(input->DFACTOR > 1) break;
	}
	(input->DFACTOR) >>= 1;
	(input->LOG_DFACTOR)--;

//	(input->DFACTOR) = 32;
//	(input->LOG_DFACTOR) = 5;

        (input->upper_np) = CEIL(nnp*(input->DFACTOR)+60,512)*512;
	//fprintf(stderr, "k : %d %d\n", input->upper_np, SM_SIZE);
	if(input->upper_np > SM_SIZE) {
		(input->DFACTOR) >>= 1;
		(input->LOG_DFACTOR)--;
        	(input->upper_np) = CEIL(nnp*(input->DFACTOR)+60,512)*512;
	}
//        (input->upper_np) = CEIL((input->np)*(input->DFACTOR)+100,512)*512;
//        (input->upper_np) = CEIL(REG_SIZE*(input->DFACTOR)+50,512)*512;

//        printf("%d %d %d %d %d\n", (input->np), input->DFACTOR, input->LOG_DFACTOR, (input->np)*(input->DFACTOR), input->upper_np);
//	printf("%d %d\n", input->nv, input->ne);

	(input->degree) = degree;

/*
	int **hist;
	int nv = (input->nv);
	int np = (input->np);
	int ne = (input->ne);
	hist = (int **)malloc(sizeof(int *)*np);
	for(i=0;i<np;i++)
		hist[i] = (int *)malloc(sizeof(int)*np);

	int j;
	for(i=0;i<np;i++)
		for(j=0;j<np;j++)
			hist[i][j]=0;

	for(i=0;i<ne;i++) {
		hist[temp[i].src/PSIZE][temp[i].dst/PSIZE]++;
	}

	exit(0);*/


        free(temp_v);
        free(loc);
        free(valid);

}

