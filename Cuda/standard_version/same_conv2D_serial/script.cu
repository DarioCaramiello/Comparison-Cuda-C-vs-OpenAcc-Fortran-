#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 10000
#define Q 3
#define K 1


void show_matrix(int*, int, int);
int* generate_sobel_kernel_x();
void same_convolution2D_version_no_sm(int*, int, int, int*, int, int, int, int*);


int main(int argn, char* argv[]) {

    // time to CPU 
    struct timespec start_time_cpu, end_time_cpu;
    struct timespec start_time_program, end_time_program;

    clock_gettime(CLOCK_MONOTONIC, &start_time_program);

    const size_t size_matrix = (N * N) * sizeof(int);
    int* matrix = (int*)malloc(size_matrix);

    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++){
            matrix[i * N + j] = i + j + 1;
        }
    }

    //printf("Matrix : \n");
    //show_matrix(matrix_h, N, N);

    // Kernel HOST
    //const size_t size_kernel = (Q * Q) * sizeof(int);
    int* kernel = generate_sobel_kernel_x();

    //printf("Kernel : \n");
    //show_matrix(kernel_h, Q, Q);


    // Result HOST
    int* result = (int*)malloc(size_matrix);

    clock_gettime(CLOCK_MONOTONIC, &start_time_cpu);
    same_convolution2D_version_no_sm(matrix, N, N, kernel, Q, Q, K, result);
    clock_gettime(CLOCK_MONOTONIC, &end_time_cpu);

    //printf("Result : \n");
    //show_matrix(result, N, N);

    free(matrix);
    //free(pad_matrix_h);
    free(kernel);
    free(result);

    clock_gettime(CLOCK_MONOTONIC, &end_time_program);

    printf("\nExecution CPU time : %f seconds.\n", (end_time_cpu.tv_sec - start_time_cpu.tv_sec) + (end_time_cpu.tv_nsec - start_time_cpu.tv_nsec) / 1000000000.0);
    printf("Execution time program : %f seconds.\n\n", (end_time_program.tv_sec - start_time_program.tv_sec) + (end_time_program.tv_nsec - start_time_program.tv_nsec) / 1000000000.0);

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



void same_convolution2D_version_no_sm(int* matrix, int rows, int cols, int* kernel, int k_rows, int k_cols, int radius, int* result) {
    int sum = 0;

    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {

            sum = 0;

            for (int r=-radius; r<=radius; r++) {
                for (int c=-radius; c<=radius; c++) {

                    int nx = r + i;
                    int ny = c + j;

                    if(nx>=0 && nx<rows && ny>=0 && ny<cols) {

                        sum += matrix[nx * cols + ny] * kernel[(r + radius) * k_cols + (c + radius)];

                    } else {

                        sum += 0;
                    }

                }

            }

            result[i * cols + j] = -sum;
        }
    }

}    


