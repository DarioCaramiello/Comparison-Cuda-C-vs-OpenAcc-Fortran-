#include <cuda.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>

#define N 500       // Matrix size
#define Q 3         // Kernel size
#define K 1         // Radius

int* generate_sobel_kernel_x();
void show_matrix(int*, int, int);
void same_convolution2D(int*, int, int, int*, int, int, int, int*);
void extract_neighborhood(int*, int, int, int, int, int, int*);
int scalar_product(int*, int*, int);

int main(int argn, char* argv[]) {

    // time to CPU 
    struct timespec start_time_cpu, end_time_cpu;
    struct timespec start_time_program, end_time_program;

    clock_gettime(CLOCK_MONOTONIC, &start_time_program);
    
    // Matrix
    int* matrix;
    const size_t size_matrix = (N * N) * sizeof(int);
    matrix = (int*)malloc(size_matrix);

    // Kernel
    int* kernel = generate_sobel_kernel_x();

    // Result Matrix 
    int* result;
    const size_t size_result = size_matrix;
    result = (int*)malloc(size_result);

    // Matrix init
    for(int i=0; i<N; i++) {
        for(int j=0; j<N; j++) {
            matrix[i * N + j] = i + j + 1;
        }
    }

    //printf("\nMatrix : \n");
    //show_matrix(matrix, N, N);
    
    //printf("Kernel : \n");
    //show_matrix(kernel, 3, 3);
    
    clock_gettime(CLOCK_MONOTONIC, &start_time_cpu);
    same_convolution2D(matrix, N, N, kernel, Q, Q, K, result);
    clock_gettime(CLOCK_MONOTONIC, &end_time_cpu);

    //printf("Result : \n");
    //show_matrix(result, N, N);

    clock_gettime(CLOCK_MONOTONIC, &end_time_program);

    printf("\nExecution CPU time : %f seconds.\n", (end_time_cpu.tv_sec - start_time_cpu.tv_sec) + (end_time_cpu.tv_nsec - start_time_cpu.tv_nsec) / 1000000000.0);
    printf("Execution time program : %f seconds.\n\n", (end_time_program.tv_sec - start_time_program.tv_sec) + (end_time_program.tv_nsec - start_time_program.tv_nsec) / 1000000000.0);

    free(matrix);
    free(kernel);
    free(result);

    return 0;
}

void show_matrix(int* matrix, int rows, int cols) {
    for(int i=0; i<rows; i++) {
        for(int j=0; j<cols; j++) {
            printf("%d\t", matrix[i * cols + j]);
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


/*
// same_convolution2D passando la matrice paddata 
void same_convolution2D(int* matrix, int rows, int cols, int* kernel, int k_rows, int k_cols, int radius, int* result){

    const int size_vector_neighborhood = ((2*radius + 1) * (2*radius + 1)) * sizeof(int);

    for(int i=radius; i<rows-radius; i++) {
        int* vector_neighborhood_ij = (int*)malloc(size_vector_neighborhood);

        for(int j=radius; j<cols-radius; j++) {
            extract_neighborhood(matrix, rows, cols, i, j, radius, vector_neighborhood_ij);            
            result[((i-radius) * N + (j-radius))] = scalar_product(vector_neighborhood_ij, kernel, ((2*radius + 1) * (2*radius + 1)));
        }

        free(vector_neighborhood_ij);
    }
    
}
*/


void same_convolution2D(int* matrix, int rows, int cols, int* kernel, int k_rows, int k_cols, int radius, int* result) {

    const int size_vector_neighborhood = ((2*radius + 1) * (2*radius + 1)) * sizeof(int);

    for(int i=0; i<rows; i++) {
        int* vector_neighborhood_ij = (int*)malloc(size_vector_neighborhood);

        for(int j=0; j<cols; j++) {
            extract_neighborhood(matrix, rows, cols, i, j, radius, vector_neighborhood_ij);
            result[(i * N + j)] = scalar_product(vector_neighborhood_ij, kernel, ((2*radius + 1) * (2*radius + 1)));

        }
    }

}


/*
// extract_neighborhood passando la matrice paddata 
void extract_neighborhood(int* matrix, int rows, int cols, int i, int j, int k, int* out) {
    
    int index_out = 0;
    
    for(int r=-k; r<=k; r++) {
        for(int c=-k; c<=k; c++) {
            out[index_out++] = matrix[(i+r) * cols + (j+c)];
        }
    }
}
*/


void extract_neighborhood(int* matrix, int rows, int cols, int i, int j, int k, int* out) {
    
    int index_out = 0;
    
    for(int r=-k; r<=k; r++) {
        for(int c=-k; c<=k; c++) {

            int nx = i + r;
            int ny = j + c;

            if(nx>=0 && nx<rows && ny>=0 && ny<cols) {
                out[index_out++] = matrix[nx * cols + ny];
            } else {
                out[index_out++] = 0;
            }

        }
    }
}

int scalar_product(int* vector, int* kernel, int size) {
    
    int scalar_out = 0;

    for(int i=0; i<size; i++) {
        scalar_out += vector[i] * kernel[i];
    }
   
    return -scalar_out;
}