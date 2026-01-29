#include "kalman_gpu.cuh"
#include <iostream>
#include <vector>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

#define CHECK_CUBLAS(call) { \
    cublasStatus_t status = call; \
    if (status != CUBLAS_STATUS_SUCCESS) { \
        printf("CUBLAS Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(1); \
    } \
}

#define CHECK_CUSOLVER(call) { \
    cusolverStatus_t status = call; \
    if (status != CUSOLVER_STATUS_SUCCESS) { \
        printf("CUSOLVER Error: %d at %s:%d\n", status, __FILE__, __LINE__); \
        exit(1); \
    } \
}

KalmanFilterGPU::KalmanFilterGPU(int state_dim, int meas_dim) : n(state_dim), m(meas_dim) {
    CHECK_CUBLAS(cublasCreate(&cublasH));
    CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_F, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_P, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Q, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_H, m * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_R, m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_x, n * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_I, n * n * sizeof(float)));

    // Intermediate buffers
    CHECK_CUDA(cudaMalloc(&d_Ft, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_FP, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_FPFt, n * n * sizeof(float)));

    CHECK_CUDA(cudaMalloc(&d_Hx, m * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_y, m * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ht, n * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_PHt, n * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_HPHt, m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_S, m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Si, m * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_K, n * m * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_Ky, n * 1 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_KH, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_I_KH, n * n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_I_KHP, n * n * sizeof(float)));

    // Workspace for cuSOLVER (getrf requires workspace)
    CHECK_CUSOLVER(cusolverDnSgetrf_bufferSize(cusolverH, m, m, d_S, m, &lwork));
    CHECK_CUDA(cudaMalloc(&d_work, lwork * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_piv, m * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
}

KalmanFilterGPU::~KalmanFilterGPU() {
    cublasDestroy(cublasH);
    cusolverDnDestroy(cusolverH);

    cudaFree(d_F); cudaFree(d_P); cudaFree(d_Q); cudaFree(d_H); cudaFree(d_R); cudaFree(d_x); cudaFree(d_I);
    cudaFree(d_Ft); cudaFree(d_FP); cudaFree(d_FPFt);
    cudaFree(d_Hx); cudaFree(d_y); cudaFree(d_Ht); cudaFree(d_PHt); cudaFree(d_HPHt);
    cudaFree(d_S); cudaFree(d_Si); cudaFree(d_K); cudaFree(d_Ky); cudaFree(d_KH); cudaFree(d_I_KH); cudaFree(d_I_KHP);
    cudaFree(d_work); cudaFree(d_piv); cudaFree(d_info);
}

void KalmanFilterGPU::setInitialState(const Matrix& F, const Matrix& P, const Matrix& Q, const Matrix& H, const Matrix& R, const Matrix& x) {
    CHECK_CUDA(cudaMemcpy(d_F, F.data.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_P, P.data.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_Q, Q.data.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_H, H.data.data(), m * n * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_R, R.data.data(), m * m * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_x, x.data.data(), n * 1 * sizeof(float), cudaMemcpyHostToDevice));

    Matrix h_I = identity(n);
    CHECK_CUDA(cudaMemcpy(d_I, h_I.data.data(), n * n * sizeof(float), cudaMemcpyHostToDevice));
}

// NOTE: cuBLAS uses column-major storage.
// Our Matrix helper and custom kernels were implementing naive row-major (or generic).
// `matrix_ops` stores data in std::vector row-by-row.
// If we copy that to device, it's row-major.
// cuBLAS Sgemm assumes column-major by default. 
// A * B in Row-Major is equivalent to B^T * A^T in Col-Major.
// OR we can tell cublas to use Transpose operations to handle row/col mismatch.
// HOWEVER, for simplicity here, I will treat everything as if it's correct and use CUBLAS_OP_T when needed to fake it, 
// OR I will just assume the data is what it is. 
// Standard trick for C (Row) -> Fortran (Col):
// C = A * B.
// In Cublas: C^T = B^T * A^T. 
// So if we store A, B, C in row-major, we can call sgemm(..., B, A) and we get C in row-major (which is C^T in col-major).
// Yes, to compute C = A * B (all row-major), call cublasSgemm(..., B, A, ...).

void KalmanFilterGPU::predict() {
    float alpha = 1.0f;
    float beta = 0.0f;

    // x = F * x
    // Row-Major: x_new (n,1) = F(n,n) * x(n,1).
    // Cublas call: Sgemm(d_x, d_F). Returns x_new^T (1,n) in col-major = x_new in row major.
    // Dim: B(n,1) * A(n,n) -> m=1, n=n, k=n. Wait.
    // C = B * A ???
    // (1, n) = (1, n) * (n, n) -> Yes. 
    // So to mult F*x in row major, we do cublasSgemm(N, 1, n, x, F).
    // wait.
    // m = N (rows of A), n = M (cols of B), k = K (cols of A / rows of B).
    // C(mxn) = A(mxk) * B(kxn).
    // In Cublas (Col Major): C = A * B.
    // To do Row Major C = A * B, we do C_col = B_col * A_col (where B_col is B data interpreted as col major).
    // Since our data IS row-major in memory, interpreting it as col-major gives us the Transpose.
    // So A_mem interpreted as col-major is A^T.
    // We want C_mem (Row) = A_mem (Row) * B_mem (Row).
    // This is equivalent to C^T (Col) = (A * B)^T (Col) = B^T (Col) * A^T (Col).
    // Since A_mem is A^T (as col), we can just multiply B_mem * A_mem.
    // So cublasSgemm(handle, OP_N, OP_N, B_cols, A_rows, A_cols, alpha, B, B_ld, A, A_ld, beta, C, C_ld).
    
    // 1. x_pred = F * x.
    // A=F(n,n), B=x(n,1). C=x_pred(n,1).
    // Call: Sgemm(..., x, F). (Order reversed).
    // M = 1 (cols of B), N = n (rows of A), K = n (cols of A).
    // lda = 1 (B's width), ldb = n (A's width), ldc = 1.
    // Effectively: X_new = x * F^T ?? No.
    // Let's verify dimensions.
    // We want C(n,1).
    // We compute C^T (1,n) = x^T (1,n) * F^T (n,n). = x^T * F^T.
    // Correct.
    // So we invoke Sgemm with M=1, N=n, K=n. A=x, B=F.
    // Wait, reusing d_x as input and output is unsafe for d_x.
    // I need a temp buffer. using d_Ky (n*1).
    
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N, 
                             1, n, n, // m, n, k
                             &alpha, 
                             d_x, n, // A is (n x 1), OP_T -> (1 x n). lda >= n.
                             d_F, n, // B is (n x n) (F^T). OP_N -> F^T.
                             &beta,
                             d_Ky, 1)); // C
    
    // Copy result back to x
    CHECK_CUDA(cudaMemcpy(d_x, d_Ky, n * 1 * sizeof(float), cudaMemcpyDeviceToDevice));
    
    // 2. P = F * P * F^T + Q
    // Step 2a: FP = F * P.
    // Row major: FP = F * P.
    // Cublas: Sgemm(P, F). M=n, N=n, K=n.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_P, n,
                             d_F, n,
                             &beta,
                             d_FP, n));
    
    // Step 2b: P_new = FP * F^T.
    // Row major. 
    // Cublas: P_new^T = (F^T)^T * FP^T = F * FP^T.
    // Wait.
    // We have FP in memory (row major). A=FP. B=F^T.
    // We want C = A * B.
    // Cublas computes C^T = B^T * A^T.
    // B^T = (F^T)^T = F.
    // A^T = FP^T.
    // So we want C^T = F * FP^T.
    // But F is in memory as F (Row). F as Col is F^T.
    // So to get F, we need Transpose(F_mem).
    // Wait, F_mem is F(row). F_mem as Col is F^T. 
    // So B^T (which is F) is F_mem interpreted as col major? No. 
    // F_mem as col major IS F^T.
    // We need F. So we need Transpose(F_mem interpreted as Col).
    // So use CUBLAS_OP_T on F_mem.
    // And A^T is FP^T. FP_mem is FP(row). FP_mem as Col is FP^T.
    // So we use OP_N on FP_mem.
    // So: Sgemm(..., F, FP) with OP_T for F?
    // Let's re-derive simply.
    // C = A * B.
    // Cublas: C = Sgemm(B, A).
    // C = FP * F^T.
    // Sgemm args: (F^T, FP).
    // But F^T is not stored physically (we have F).
    // B = F^T. A = FP.
    // Call Sgemm(B, A).
    // B is F^T.
    // We have F.
    // F_mem is F.
    // We can pass F_mem and say OP_T?
    // If we pass F_mem with OP_N -> it sees F^T.
    // If we pass F_mem with OP_T -> it sees (F^T)^T = F.
    // BUT we want B = F^T.
    // So we passed "F^T".
    // Wait. 
    // If we want B to be F^T.
    // And we have F in memory.
    // F in memory is F^T (in col view).
    // So F in memory IS ALREADY B (in col view).
    // So we pass F with OP_N.
    // Let's check:
    // We want C = FP * F^T.
    // Call Sgemm params: B=(F^T), A=FP.
    // Param 1 (B form): We pass d_F. CuBlas sees d_F as F^T. This matches B. Good. OP_N.
    // Param 2 (A form): We pass d_FP. CuBlas sees d_FP as FP^T. This matches A^T?
    // Wait. 
    // C^T = B^T * A^T.
    // We want C = FP * F^T.
    // So C^T = (FP * F^T)^T = (F^T)^T * FP^T = F * FP^T.
    // We need to compute F * FP^T.
    // Matrix 1: F. (We have d_F which is F^T). So we need OP_T on d_F to get F.
    // Matrix 2: FP^T. (We have d_FP which is FP^T). So we need OP_N on d_FP.
    // So: Sgemm(..., d_FP, d_F) with OP_N for FP and OP_T for F.
    // Dimensions: M (rows of 1st mat in op), N (cols of 2nd), K.
    // Result is C^T (n x n).
    // F (n x n) * FP^T (n x n).
    // M=n, N=n, K=n.
    // So: Sgemm(handle, OP_N, OP_T, n, n, n, alpha, d_FP, n, d_F, n, beta, d_P, n).
    
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_T,
                             n, n, n,
                             &alpha,
                             d_FP, n,
                             d_F, n,
                             &beta,
                             d_P, n));

    // Step 2c: P = P + Q
    // Use custom kernel for addition as cublasGeam is overkill or same.
    // Or use cublasSaxpy if we treat as vector?
    // Yes, P and Q are continuous vectors.
    // P = P + 1*Q.
    CHECK_CUBLAS(cublasSaxpy(cublasH, n*n, &alpha, d_Q, 1, d_P, 1));
}

void KalmanFilterGPU::update(const Matrix& z) {
    float alpha = 1.0f;
    float minus_one = -1.0f;
    float zero = 0.0f;

    CHECK_CUDA(cudaMemcpy(d_y, z.data.data(), m * 1 * sizeof(float), cudaMemcpyHostToDevice));

    // 1. y = z - Hx
    // Hx = H * x.
    // Row major C = A * B. -> Sgemm(B, A).
    // B=x, A=H.
    // M=1, N=m, K=n.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                             1, m, n,
                             &alpha,
                             d_x, n, // lda=n
                             d_H, n, // lda is cols of A -> n. B is m x n (H). Viewed as n x m (H^T). rows=n.
                             &zero,
                             d_Hx, 1));
    
    // y = y - Hx.
    // d_y has z. d_Hx has Hx.
    // y = 1*y - 1*Hx.
    CHECK_CUBLAS(cublasSaxpy(cublasH, m, &minus_one, d_Hx, 1, d_y, 1));


    // 2. S = H * P * H^T + R
    // PHt = P * H^T.
    // C = A * B. Sgemm(B, A).
    // B=H^T. A=P.
    // B^T = H. We have d_H (H^T view). So B^T is d_H.
    // We want B^T * A^T. H * P^T.
    // We have d_H (H^T). OP_T -> H.
    // We have d_P (P^T). OP_N -> P^T.
    // So Sgemm(..., d_P, d_H) with OP_N, OP_T.
    // M=n (rows of H^T aka cols of H -> n? No wait).
    // Result PHt is (n x m).
    // Transpose result: (m x n).
    // We compute K=n.
    // Wait. 
    // P (n x n) * H^T (n x m).
    // Result (n x m).
    // Sgemm computes Result^T (m x n).
    // Result^T = (P * H^T)^T = H * P^T.
    // H (m x n) * P^T (n x n).
    // M=m, N=n, K=n.
    // Op1: H. (d_H is H^T). Use OP_T.
    // Op2: P^T. (d_P is P^T). Use OP_N.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                             m, n, n, 
                             &alpha,
                             d_H, n, // A: H (via OP_T on d_H)
                             d_P, n, // B: P^T (via OP_N on d_P)
                             &zero,
                             d_PHt, m)); // C: result m x n
    
    // HPHt = H * PHt.
    // C = A * B.
    // Sgemm(B, A).
    // B = PHt. A = H.
    // Res (m x m). Res^T (m x m).
    // Res^T = PHt^T * H^T.
    // We have d_PHt (PHt^T view). So OP_N.
    // We have d_H (H^T view). So OP_N.
    // M=m, N=m, K=n.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, m, n,
                             &alpha,
                             d_PHt, m,
                             d_H, n,
                             &zero,
                             d_HPHt, m));

    // S = HPHt + R.
    CHECK_CUDA(cudaMemcpy(d_S, d_R, m * m * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS(cublasSaxpy(cublasH, m*m, &alpha, d_HPHt, 1, d_S, 1));


    // 3. K = PHt * S^-1
    // We need S^-1.
    // Using cusolverDnSgetrf + cusolverDnSgetrs (Linear Solver) or getri (Inverse).
    // Since we compute K = PHt * S^-1.
    // Let's compute S^-1 explicitly using getrf + getri (General Inverse).
    // d_S is input.
    
    // LU Factorization
    CHECK_CUSOLVER(cusolverDnSgetrf(cusolverH, m, m, d_S, m, d_work, d_piv, d_info));
    // Valid check?
    // int info_h; cudaMemcpy(&info_h, d_info, sizeof(int), cudaMemcpyDeviceToHost); if(info_h!=0) printf("Singular\n");

    // Inverse using LU + Solve
    // We want S^-1.
    // Solve S * X = I.  => X = S^-1.
    // d_S contains LU factors from getrf.
    // We need 'Right hand side' B = I.
    // Result X will be stored in B.
    // So we copy Identity to d_Si first.
    CHECK_CUDA(cudaMemcpy(d_Si, d_I, m * m * sizeof(float), cudaMemcpyDeviceToDevice)); // Assumption: d_I has m*m Identity? No, d_I is n*n.
    // We need an m*m Identity.
    // Let's create it on the fly or assumes we have one.
    // Helper: We can just use a kernel to set d_Si to Identity or copy from host.
    {
        std::vector<float> h_I_m(m*m, 0.0f);
        for(int i=0; i<m; ++i) h_I_m[i*m + i] = 1.0f;
        CHECK_CUDA(cudaMemcpy(d_Si, h_I_m.data(), m*m*sizeof(float), cudaMemcpyHostToDevice));
    }

    // Now solve S * X = Si (where Si is I). Result placed in Si.
    CHECK_CUSOLVER(cusolverDnSgetrs(cusolverH, CUBLAS_OP_N, m, m, d_S, m, d_piv, d_Si, m, d_info));
    // Now d_Si contains S^-1.

    // K = PHt * S^-1.
    // C = A * B. Sgemm(B,A).
    // B = S^-1. A = PHt.
    // Res (n x m). Res^T (m x n).
    // Res^T = (S^-1)^T * PHt^T.
    // S^-1 is symmetric? Usually yes for Kalman. So (S^-1)^T = S^-1.
    // d_Si has S^-1 (col major view = row major view if symmetric).
    // d_PHt has PHt^T view.
    // So Sgemm(d_Si, d_PHt). OP_N, OP_N.
    // M=m, N=n, K=m.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             m, n, m,
                             &alpha,
                             d_Si, m,
                             d_PHt, m,
                             &zero,
                             d_K, m));

    // 4. x = x + K * y
    // Ky = K * y.
    // Sgemm(y, K).
    // Res (n x 1). Res^T (1 x n).
    // Res^T = y^T * K^T.
    // d_y (y^T view). d_K (K^T view).
    // OP_N, OP_N.
    // M=1, N=n, K=m.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_T, CUBLAS_OP_N,
                             1, n, m,
                             &alpha,
                             d_y, m, // lda=m
                             d_K, m,
                             &zero,
                             d_Ky, 1));
    
    // x = x + Ky
    CHECK_CUBLAS(cublasSaxpy(cublasH, n, &alpha, d_Ky, 1, d_x, 1));


    // 5. P = (I - KH) * P
    // KH = K * H.
    // Sgemm(H, K).
    // Res (n x n). Res^T (n x n).
    // Res^T = H^T * K^T.
    // d_H (H^T view). d_K (K^T view).
    // M=n, N=n, K=m.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, m,
                             &alpha,
                             d_H, n,
                             d_K, m,
                             &zero,
                             d_KH, n));
    
    // I - KH.
    CHECK_CUDA(cudaMemcpy(d_I_KH, d_I, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
    CHECK_CUBLAS(cublasSaxpy(cublasH, n*n, &minus_one, d_KH, 1, d_I_KH, 1));
    
    // P_new = (I-KH) * P.
    // Sgemm(P, I-KH).
    // Res^T = P^T * (I-KH)^T.
    // d_P (P^T view). d_I_KH ( (I-KH)^T view).
    // M=n, N=n, K=n.
    CHECK_CUBLAS(cublasSgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N,
                             n, n, n,
                             &alpha,
                             d_P, n,
                             d_I_KH, n,
                             &zero,
                             d_I_KHP, n));
    
    CHECK_CUDA(cudaMemcpy(d_P, d_I_KHP, n * n * sizeof(float), cudaMemcpyDeviceToDevice));
}

void KalmanFilterGPU::getState(Matrix& x_out) {
    CHECK_CUDA(cudaMemcpy(x_out.data.data(), d_x, n * 1 * sizeof(float), cudaMemcpyDeviceToHost));
}
