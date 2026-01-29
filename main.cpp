#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include "matrix_ops.h"
#include "kalman_cpu.h"
#include "kalman_gpu.cuh"

// Helper to generate synthetic measurements
std::vector<Matrix> generateMeasurements(int m, int count) {
    std::vector<Matrix> measurements;
    for (int i = 0; i < count; ++i) {
        Matrix z(m, 1);
        for (int j = 0; j < m; ++j) {
            z(j, 0) = static_cast<float>(i + j * 0.1); 
        }
        measurements.push_back(z);
    }
    return measurements;
}

void run_benchmark(int n, int m, int steps, bool check_correctness = false) {
    std::cout << "\n==============================================" << std::endl;
    std::cout << "Benchmarking N=" << n << ", M=" << m << ", Steps=" << steps << std::endl;
    
    // Initialization
    Matrix F(n, n);
    for (int i=0; i<n; ++i) F(i, i) = 1.0f; // Identity transition
    
    Matrix P(n, n);
    for (int i=0; i<n; ++i) P(i, i) = 1.0f;

    Matrix Q(n, n);
    for (int i=0; i<n; ++i) Q(i, i) = 0.01f;

    Matrix H(m, n);
    for (int i=0; i<m; ++i) H(i, i % n) = 1.0f; // Simple H

    Matrix R(m, m);
    for (int i=0; i<m; ++i) R(i, i) = 0.1f;

    Matrix x(n, 1); // Initial state zero

    auto measurements = generateMeasurements(m, steps);

    // --- CPU Run ---
    KalmanFilterCPU kf_cpu(n, m);
    kf_cpu.F = F; kf_cpu.P = P; kf_cpu.Q = Q; kf_cpu.H = H; kf_cpu.R = R; kf_cpu.x = x;

    std::cout << "Running CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for (const auto& z : measurements) {
        kf_cpu.predict();
        kf_cpu.update(z);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    std::cout << "CPU Time: " << elapsed_cpu.count() << " ms" << std::endl;

    // --- GPU Run (Optimized w/ cuBLAS) ---
    KalmanFilterGPU kf_gpu(n, m);
    kf_gpu.setInitialState(F, P, Q, H, R, x);

    // Warmup
    {
       Matrix z_temp(m, 1);
       kf_gpu.predict();
       kf_gpu.update(z_temp);
       kf_gpu.setInitialState(F, P, Q, H, R, x); // Reset
       cudaDeviceSynchronize();
    }

    std::cout << "Running GPU..." << std::endl;
    auto start_gpu = std::chrono::high_resolution_clock::now();
    for (const auto& z : measurements) {
        kf_gpu.predict();
        kf_gpu.update(z);
    }
    cudaDeviceSynchronize(); // Ensure completion
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_gpu = end_gpu - start_gpu;
    std::cout << "GPU Time: " << elapsed_gpu.count() << " ms" << std::endl;
    std::cout << "Speedup: " << elapsed_cpu.count() / elapsed_gpu.count() << "x" << std::endl;

    // --- Verification ---
    if (check_correctness) {
        Matrix cpu_final_x = kf_cpu.x;
        Matrix gpu_final_x(n, 1);
        kf_gpu.getState(gpu_final_x);

        float mse = 0.0f;
        for (int i = 0; i < n; ++i) {
            float diff = cpu_final_x(i, 0) - gpu_final_x(i, 0);
            mse += diff * diff;
        }
        float rmse = std::sqrt(mse / n);

        std::cout << "RMSE between CPU and GPU results: " << rmse << std::endl;
        if (rmse < 1e-2) { // Looser tolerance for float vs double or precision diffs
            std::cout << "Validation: PASSED" << std::endl;
        } else {
            std::cout << "Validation: FAILED" << std::endl;
        }
    }
}

int main() {
    // 1. Tiny Scale (N=2)
    run_benchmark(2, 2, 1000, true);

    // 2. Small Scale (N=4)
    run_benchmark(4, 2, 1000, true);

    // 3. Medium Scale (N=50)
    run_benchmark(50, 50, 1000, true);

    // 4. Large Scale (N=300)
    // Note: CPU complexity increases cubicly. N=300 might take a minute or two on CPU.
    run_benchmark(300, 300, 100, true);
    
    return 0;
}
