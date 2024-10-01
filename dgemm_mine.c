#define _XOPEN_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>
#include <mmintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>

const char* dgemm_desc = "Three-level blocked dgemm with AVX2 and manual loop unrolling.";

#ifndef BLOCK_SIZE_L3
#define BLOCK_SIZE_L3 ((int) 512)
#endif

#ifndef BLOCK_SIZE_L2
#define BLOCK_SIZE_L2 ((int) 128)
#endif

#ifndef BLOCK_SIZE_L1
#define BLOCK_SIZE_L1 ((int) 32)
#endif

// Align temporary arrays to 32 bytes for AVX2
static __attribute__((aligned(32))) double a[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
static __attribute__((aligned(32))) double b[BLOCK_SIZE_L1*BLOCK_SIZE_L1];
static __attribute__((aligned(32))) double c[BLOCK_SIZE_L1*BLOCK_SIZE_L1];

void kernel_dgemm(const int lda, const int M, const int N, const int K, 
                  const double * restrict A , const double * restrict B, double * restrict C)
{
 
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            a[i + k * BLOCK_SIZE_L1] = A[i + k * lda];
        }
    }

    for (int k = 0; k < K; k++) {
        for (int j = 0; j < N; j++) {
            b[k + j * BLOCK_SIZE_L1] = B[k + j * lda];
        }
    }

    // Vectorized loop using AVX2
    for (int j = 0; j < BLOCK_SIZE_L1; ++j) {
        for (int k = 0; k < BLOCK_SIZE_L1; ++k) {
            __m256d b_vec = _mm256_set1_pd(b[k + j * BLOCK_SIZE_L1]);  // Broadcast B[k][j]

            for (int i = 0; i < BLOCK_SIZE_L1; i += 4) {  // Unroll the inner loop: process 4 elements at a time
                __m256d c_vec = _mm256_load_pd(&c[j * BLOCK_SIZE_L1 + i]);  // Load C[i:i+4, j]
                __m256d a_vec = _mm256_load_pd(&a[k * BLOCK_SIZE_L1 + i]);  // Load A[i:i+4, k]

                // Perform fused multiply-add (C += A * B)
                c_vec = _mm256_fmadd_pd(a_vec, b_vec, c_vec);

                // Store the result back to C
                _mm256_store_pd(&c[j * BLOCK_SIZE_L1 + i], c_vec);
            }
        }
    }

    // Write back the results from temporary storage to C
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            C[j * lda + i] += c[j * BLOCK_SIZE_L1 + i];
        }
    }

    // Clear temporary arrays
    memset(c, 0, sizeof(double) * BLOCK_SIZE_L1 * BLOCK_SIZE_L1);
    memset(a, 0, sizeof(double) * BLOCK_SIZE_L1 * BLOCK_SIZE_L1);
    memset(b, 0, sizeof(double) * BLOCK_SIZE_L1 * BLOCK_SIZE_L1);
}

// L1 blocking
void do_block_l1(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE_L1 > lda) ? (lda - i) : BLOCK_SIZE_L1;
    const int N = (j + BLOCK_SIZE_L1 > lda) ? (lda - j) : BLOCK_SIZE_L1;
    const int K = (k + BLOCK_SIZE_L1 > lda) ? (lda - k) : BLOCK_SIZE_L1;
    kernel_dgemm(lda, M, N, K, A + i + k * lda, B + k + j * lda, C + i + j * lda);
}

// L2 blocking
void do_block_l2(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE_L2 > lda) ? (lda - i) : BLOCK_SIZE_L2;
    const int N = (j + BLOCK_SIZE_L2 > lda) ? (lda - j) : BLOCK_SIZE_L2;
    const int K = (k + BLOCK_SIZE_L2 > lda) ? (lda - k) : BLOCK_SIZE_L2;
    
    for (int bk = 0; bk < K; bk += BLOCK_SIZE_L1) {  
        for (int bj = 0; bj < N; bj += BLOCK_SIZE_L1) {  
            for (int bi = 0; bi < M; bi += BLOCK_SIZE_L1) {  
                do_block_l1(lda, A, B, C, i + bi, j + bj, k + bk);
            }
        }
    }
}

// L3 blocking
void do_block_l3(const int lda, 
                 const double *A, const double *B, double *C,
                 const int i, const int j, const int k)
{
    const int M = (i + BLOCK_SIZE_L3 > lda) ? (lda - i) : BLOCK_SIZE_L3;
    const int N = (j + BLOCK_SIZE_L3 > lda) ? (lda - j) : BLOCK_SIZE_L3;
    const int K = (k + BLOCK_SIZE_L3 > lda) ? (lda - k) : BLOCK_SIZE_L3;
    
    for (int bk = 0; bk < K; bk += BLOCK_SIZE_L2) {  
        for (int bj = 0; bj < N; bj += BLOCK_SIZE_L2) { 
            for (int bi = 0; bi < M; bi += BLOCK_SIZE_L2) {  
                do_block_l2(lda, A, B, C, i + bi, j + bj, k + bk);
            }
        }
    }
}

// Top-level function
void square_dgemm(const int M, 
                  const double * restrict A, 
                  const double * restrict B, 
                  double * restrict C)
{
    for (int k = 0; k < M; k += BLOCK_SIZE_L3) {  
        for (int j = 0; j < M; j += BLOCK_SIZE_L3) {  
            for (int i = 0; i < M; i += BLOCK_SIZE_L3) {  
                do_block_l3(M, A, B, C, i, j, k);
            }
        }
    }
}
