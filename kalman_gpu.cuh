#ifndef KALMAN_GPU_CUH
#define KALMAN_GPU_CUH

#include <cublas_v2.h>
#include <cusolverDn.h>
#include "matrix_ops.h" 

class KalmanFilterGPU {
public:
    int n; // state dimension
    int m; // measurement dimension

    cublasHandle_t cublasH;
    cusolverDnHandle_t cusolverH;

    // Device pointers
    float *d_F, *d_P, *d_Q, *d_H, *d_R, *d_x, *d_I;
    
    // Intermediate buffers (Device)
    float *d_Ft, *d_FP, *d_FPFt;
    float *d_Hx, *d_y, *d_Ht, *d_PHt, *d_HPHt, *d_S, *d_Si, *d_K, *d_Ky, *d_KH, *d_I_KH, *d_I_KHP;

    // Workspace for cuSOLVER
    float *d_work;
    int *d_piv; // Pivot info for LU
    int *d_info; // Info for LAPACK
    int lwork; 

    KalmanFilterGPU(int state_dim, int meas_dim);
    ~KalmanFilterGPU();

    void setInitialState(const Matrix& F, const Matrix& P, const Matrix& Q, const Matrix& H, const Matrix& R, const Matrix& x);
    void predict();
    void update(const float* d_z); // z is already on device or we copy it? Let's assume z passed as device ptr for efficiency if possible, or host. Let's take host vector for simplicity in main loop, or device ptr. I'll take host array for simplicity of API.
    void update(const Matrix& z); // Helper that copies z to device

    void getState(Matrix& x); // Copy back to host
};

#endif // KALMAN_GPU_CUH
