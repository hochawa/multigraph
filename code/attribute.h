#if defined BFS

#define GRAPHTYPE struct HYB
#define V1T int
#define INT_T

#endif

#if defined SSSP

#define GRAPHTYPE struct HYB
#define V1T int
#define E1 // edge attribute1
#define E1T int // type of edge attribute1
#define INT_T

#endif

#if defined BC

#define GRAPHTYPE struct HYB
#define V1T int
#define V2
#define V2T int
#define V3
#define V3T float
#define INT_T

#endif

#if defined CC

#define GRAPHTYPE struct MULTI_SPARSE 
#define V1T int
#define V2
#define V2T int
#define INT_T

#endif

#if defined PR_T

#define GRAPHTYPE struct HYB
#define V1T float
#define FLOAT_T

#endif

#if defined PR_D || defined PR_T

#define GRAPHTYPE struct HYB
#define V1T float
#define V2
#define V2T int
#define FLOAT_T

#endif



#ifndef E1
#define E1T int // assume edge type is "INT"
#endif
 
