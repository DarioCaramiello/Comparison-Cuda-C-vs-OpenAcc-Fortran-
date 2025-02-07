#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

// -- test 1 
#define N 5    // Matrix size
#define Q 3         // Kernel size
#define K 1         // Radius

void show_matrix(int*, int, int);
int* generate_sobel_kernel_x();
__global__ void same_convolution2D(int*, int, int, int*, int, int, int, int*);


int main(int argv, char** argc) {

    // variabili per misurare solo il nucleo computazionale 
    cudaEvent_t start_time_kernel, end_time_kernel;
    float elapsed_time_kernel;
    cudaEventCreate(&start_time_kernel);
    cudaEventCreate(&end_time_kernel);

    // variabili per misurare il passaggio di memoria + nucleo computazionale ( calcolabile anche come : elapsedTime_GPU_comp + (elapsedTime_memory_GPU_1 + elapsedTime_memory_GPU_2))
    cudaEvent_t start_time_program, end_time_program;
    float elapsed_time_program;
    cudaEventCreate(&start_time_program);
    cudaEventCreate(&end_time_program);

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
    
    cudaEventRecord(start_time_program, 0);

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
    const size_t size_s_m = (((nThreadsPerBlock.x * nThreadsPerBlock.y)) * ((2*K+1) * (2*K+1))) * sizeof(int);

    //printf("\n--------------------------------------------------\n");
    //printf("-           Grid Configuration                     -\n");
    //printf("--------------------------------------------------\n");
    //printf("- Size grid (matrix of blocks ) : [ %d ][ %d ]   -\n", nBlocks.x, nBlocks.y);
    //printf("- Size blocks (matrix of threads ) : [ %d ][ %d ]-\n", nThreadsPerBlock.x, nThreadsPerBlock.y);

    cudaEventRecord(start_time_kernel,0);
    same_convolution2D<<<nBlocks, nThreadsPerBlock, size_s_m>>>(matrix_d, N, N, kernel_d, Q, Q, K, result_d);
    cudaEventRecord(end_time_kernel, 0);
    cudaEventSynchronize(end_time_kernel);

    cudaEventRecord(start_memory_d_time_2, 0);
    cudaMemcpy(result_h, result_d, size_matrix, cudaMemcpyDeviceToHost);
    cudaEventRecord(end_memory_d_time_2, 0);
    cudaEventSynchronize(end_memory_d_time_2);

    printf("Result Matrix : \n");
    show_matrix(result_h, N, N);

    cudaEventRecord(end_time_program, 0);
    cudaEventSynchronize(end_time_program);

    cudaEventElapsedTime(&elapsed_time_kernel, start_time_kernel, end_time_kernel);
    cudaEventElapsedTime(&elapsed_time_program, start_time_program, end_time_program);

    cudaEventElapsedTime(&elapsedTime_memory_GPU_1, start_memory_d_time_1, end_memory_d_time_1);
    cudaEventElapsedTime(&elapsedTime_memory_GPU_2, start_memory_d_time_2, end_memory_d_time_2);

    printf("\nExecution kernel time: %f seconds.\n", elapsed_time_kernel / 1000.0);
    printf("Execution program time: %f seconds.\n", elapsed_time_program / 1000.0);
    printf("Execution memory time (host --> device)(device --> host) : %f seconds.\n\n", (elapsedTime_memory_GPU_1 + elapsedTime_memory_GPU_2) / 1000.0);

    free(matrix_h);
    free(kernel_h);
    free(result_h);

    cudaFree(matrix_d);
    cudaFree(kernel_d);
    cudaFree(result_d);

    return 0;
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


__global__ void same_convolution2D(int* matrix, int rows, int cols, int* kernel, int k_rows, int k_cols, int radius, int* result) {

    extern __shared__ int neighbors[];
    
    const int size_window = (2*radius+1) * (2*radius+1);

    int index_x = threadIdx.y + blockIdx.y * blockDim.y;
    int index_y = threadIdx.x + blockIdx.x * blockDim.x;

    if(index_x>=0 && index_x<rows && index_y>=0 && index_y<cols) {


        //int index_row_sm = index_x * cols + index_y;
        int index_row_sm = threadIdx.y * 16 + threadIdx.x;

        int count = 0;

        for(int i=-radius; i<=radius; i++) {
            for(int j=-radius; j<=radius; j++) {

                int nx = index_x + i;
                int ny = index_y + j;

                int val_tmp = 0;

                if(nx>=0 && nx < rows && ny>=0 && ny<cols) {
                    val_tmp = matrix[nx * cols + ny];
                } else {
                    val_tmp = 0;
                }


                neighbors[index_row_sm * size_window + count] = -val_tmp;
                count+=1;
            }
        }

        __syncthreads();

        //printf("Thread [%d][%d] : %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t %d\t  \n", index_x, index_y, neighbors[index_row_sm * size_window],
        //                                                                                   neighbors[index_row_sm * size_window+1],
        //                                                                                   neighbors[index_row_sm * size_window+2],
        //                                                                                   neighbors[index_row_sm * size_window+3],
        //                                                                                   neighbors[index_row_sm * size_window+4],
        //                                                                                   neighbors[index_row_sm * size_window+5],
        //                                                                                   neighbors[index_row_sm * size_window+6],
        //                                                                                   neighbors[index_row_sm * size_window+7],
        //                                                                                   neighbors[index_row_sm * size_window+8]);
        
        int sum = 0;
        count = 0;



        for(int i=0; i<(2*radius+1); i++) {
            for(int j=0; j<(2*radius+1); j++) {
                sum += neighbors[index_row_sm * size_window + count] * kernel[i * k_cols + j];
                count++;
            }
        }

        result[index_x * cols + index_y] = sum;

    }

}