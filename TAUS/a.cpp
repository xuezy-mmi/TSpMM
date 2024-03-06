#include <stdio.h>
#include <stdlib.h>

int main(int argc, char * argv[])
{
    if (argc != 2){
        printf("input error. please input ./a.out X\n");
        exit(1);
    }
    const int BM = 8;
    const int BK = 32;
    const int BN = 64;
    int nnz = std::atoi(argv[1]);
// main loop
    int cnnz = nnz;
    printf("nnz = %d, cnnz = %d\n", nnz, cnnz);
    for (; nnz > 0; nnz -= BK){
        printf("\nLOAD_32, nnz = %d\n", nnz);

        if(nnz <= BK) break;
        
        printf("\nCOMPUTE_32\n");
        cnnz = cnnz - BK;
    }
    printf("\nnnz = %d, cnnz = %d\n\n", nnz, cnnz);
// residue
    for(int i = 0; i < 4; i++){
        printf("\nCOMPUTE_8, cnnz = %d\n", cnnz);
        cnnz -= 8;
        if(cnnz <= 0) break;
    }
    printf("\nnnz = %d, cnnz = %d\n", nnz, cnnz);
    return 0;
}