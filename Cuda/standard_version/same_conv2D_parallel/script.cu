#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 5
#define Q 3
#define K 1

void show_matrix(int*, int, int);
int* generate_sobel_kernel_x();
__global__ void same_convolution2D_version_no_sm(int*, int, int, int*, int, int, int, int*);


int main(int argn, char* argv[]) {

    // variabili per misurare solo il nucleo computazionale 
    cudaEvent_t start_comp, end_comp;
    float elapsedTime_GPU_comp;
    cudaEventCreate(&start_comp);
    cudaEventCreate(&end_comp);

    // variabili per misurare il passaggio di memoria + nucleo computazionale ( calcolabile anche come : elapsedTime_GPU_comp + (elapsedTime_memory_GPU_1 + elapsedTime_memory_GPU_2))
    cudaEvent_t start_all, end_all;
    float elapsedTime_GPU_all;
    cudaEventCreate(&start_all);
    cudaEventCreate(&end_all);

    // variabili per misurare l'allocazione e passaggio del kernel e matrix e l'allocazione del result
    // passaggio metrix, kernel --> host to device
    // allocazione result --> device 
    cudaEvent_t start_memory_d_time_1, end_memory_d_time_1;
    float elapsedTime_memory_GPU_1;
    cudaEventCreate(&start_memory_d_time_1);
    cudaEventCreate(&end_memory_d_time_1);

    // variabili per misurare il passaggio del result da device ad host 
    // passaggio result --> device to host 
    cudaEvent_t start_memory_d_time_2, end_memory_d_time_2;
    float elapsedTime_memory_GPU_2;
    cudaEventCreate(&start_memory_d_time_2);
    cudaEventCreate(&end_memory_d_time_2);





    // Matrix HOST
    const size_t size_matrix = (N * N) * sizeof(int);
    int* matrix_h = (int*)malloc(size_matrix);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++){
            matrix_h[i * N + j] = i + j + 1;
        }
    }

    
    // Kernel HOST
    const size_t size_kernel = (Q * Q) * sizeof(int);
    int* kernel_h = generate_sobel_kernel_x();



    // Result HOST
    int* result_h = (int*)malloc(size_matrix);

    //printf("Matrix : \n");
    //show_matrix(matrix_h, N, N);

    //printf("Kernel : \n");
    //show_matrix(kernel_h, Q, Q);


    cudaEventRecord(start_all, 0);


    cudaEventRecord(start_memory_d_time_1, 0);

    // Matrix DEVICE
    int* matrix_d;
    cudaMalloc((void**)&matrix_d, size_matrix);
    cudaMemcpy(matrix_d, matrix_h, size_matrix, cudaMemcpyHostToDevice);

    // Kernel DEVICE
    int* kernel_d;
    cudaMalloc((void**)&kernel_d, size_kernel);
    cudaMemcpy(kernel_d, kernel_h, size_kernel, cudaMemcpyHostToDevice);

    //Result DEVICE
    int* result_d;
    cudaMalloc((void**)&result_d, size_matrix);

    cudaEventRecord(end_memory_d_time_1, 0);
    cudaEventSynchronize(end_memory_d_time_1);

    dim3 nThreadsPerBlock(16, 16);
    dim3 nBlocks((N + nThreadsPerBlock.x - 1) / nThreadsPerBlock.x, (N + nThreadsPerBlock.y - 1) / nThreadsPerBlock.y);
   
    //printf("\n--------------------------------------------------\n");
    //printf("-           Configurazione Griglia               -\n");
    //printf("--------------------------------------------------\n");
    //printf("- Size grid (matrix of blocks ) : [ %d ][ %d ]   -\n", nBlocks.x, nBlocks.y);
    //printf("- Size blocks (matrix of threads ) : [ %d ][ %d ]-\n", nThreadsPerBlock.x, nThreadsPerBlock.y);
    //printf("--------------------------------------------------\n\n");

    cudaEventRecord(start_comp, 0);
    same_convolution2D_version_no_sm<<<nBlocks, nThreadsPerBlock>>>(matrix_d, N, N, kernel_d, Q, Q, K, result_d);
    cudaEventRecord(end_comp, 0);
    cudaEventSynchronize(end_comp);

    cudaEventRecord(start_memory_d_time_2, 0);
    
    cudaMemcpy(result_h, result_d, size_matrix, cudaMemcpyDeviceToHost);

    cudaEventRecord(end_memory_d_time_2, 0);
    cudaEventSynchronize(end_memory_d_time_2);

    cudaEventRecord(end_all, 0);
    cudaEventSynchronize(end_all);

    printf("Result : \n");
    show_matrix(result_h, N, N);

    cudaEventElapsedTime(&elapsedTime_GPU_comp, start_comp, end_comp);
    cudaEventElapsedTime(&elapsedTime_GPU_all, start_all, end_all);

    cudaEventElapsedTime(&elapsedTime_memory_GPU_1, start_memory_d_time_1, end_memory_d_time_1);
    cudaEventElapsedTime(&elapsedTime_memory_GPU_2, start_memory_d_time_2, end_memory_d_time_2);


    printf("Execution kernel time : %f seconds.\n", elapsedTime_GPU_comp / 1000.0);
    printf("Execution program time : %f seconds.\n", elapsedTime_GPU_all / 1000.0);
    printf("Execution memory time (host --> device) (device --> host) : %f seconds.\n\n", (elapsedTime_memory_GPU_1 + elapsedTime_memory_GPU_2) / 1000.0);

    free(matrix_h);
    //free(pad_matrix_h);
    free(kernel_h);
    free(result_h);

    cudaFree(matrix_d);
    //cudaFree(pad_matrix_d);
    cudaFree(kernel_d);
    cudaFree(result_d);

    exit(1);
}

void show_matrix(int* matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%d \t", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int* generate_sobel_kernel_x() {
    int* kernel = (int*)malloc(9 * sizeof(int));
    
    int sobel_x[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1}
    };

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            kernel[i * 3 + j] = sobel_x[i][j];
        }
    }

    return kernel;
}



__global__ void same_convolution2D_version_no_sm(int* matrix, int rows, int cols, int* kernel, int k_rows, int k_cols, int radius, int* result) {

    // in questo modo gli indici globali dei thread gestiranno solo la matrice 4x4 (matrix senza padding) invece che 5x5
    // di fatti sono sicuro che x e y , per com'è configurata la griglia dei thread (rowsXcols) , saranno indici che rientreranno nella matrice
    int g_x = blockIdx.y * blockDim.y + threadIdx.y;
    int g_y = blockIdx.x * blockDim.x + threadIdx.x;

    int sum = 0;
    
    if(g_x>=0 && g_x<rows && g_y>=0 && g_y<cols) {

        sum = 0;
        // --- indici dei vicini in un raggio di k ---
        // scorre i vicini , funziona perche x e y partono da 0+k per le righe (k = padding = 1) e 0+k per le colonne
        for (int i = -radius; i <= radius; i++) {
            for (int j = -radius; j <= radius; j++) {

                int nx = g_x + i;
                int ny = g_y + j;

                if(nx>=0 && nx<rows && ny>=0 && ny<cols) {

                    sum += matrix[nx * cols + ny] * kernel[(i + radius) * k_cols + (j + radius)];

                } else {

                    sum += 0;
                }

            }
        }

        //__syncthreads();

        // --- (x-k) = i senza padding ----
        // --- (y-k) = j senza padding ----
        // --- (cols_matrix-2*k) size senza padding ----
        // result[(g_x-radius) * (cols-2*radius) + (g_y-radius)] = -sum;
        result[g_x * cols + g_y] = -sum;
    }
}    


