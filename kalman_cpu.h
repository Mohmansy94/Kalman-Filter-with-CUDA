#ifndef KALMAN_CPU_H
#define KALMAN_CPU_H

#include "matrix_ops.h"

class KalmanFilterCPU {
public:
    int n; // state dimension
    int m; // measurement dimension

    Matrix F; // State transition matrix (n x n)
    Matrix P; // Covariance matrix (n x n)
    Matrix Q; // Process noise covariance (n x n)
    Matrix H; // Measurement matrix (m x n)
    Matrix R; // Measurement noise covariance (m x m)
    Matrix x; // State vector (n x 1)
    Matrix I; // Identity matrix (n x n)

    KalmanFilterCPU(int state_dim, int meas_dim) 
        : n(state_dim), m(meas_dim), 
          F(n, n), P(n, n), Q(n, n), H(m, n), R(m, m), x(n, 1), I(identity(n)) {}

    void predict() {
        // x = F * x
        x = mult(F, x);
        
        // P = F * P * F^T + Q
        Matrix Ft = transpose(F);
        P = add(mult(mult(F, P), Ft), Q);
    }

    void update(const Matrix& z) {
        // y = z - H * x
        Matrix Hx = mult(H, x);
        Matrix y = sub(z, Hx);

        // S = H * P * H^T + R
        Matrix Ht = transpose(H);
        Matrix PHt = mult(P, Ht);
        Matrix S = add(mult(H, PHt), R);

        // K = P * H^T * S^-1
        Matrix Si = inverse(S);
        Matrix K = mult(PHt, Si);

        // x = x + K * y
        x = add(x, mult(K, y));

        // P = (I - K * H) * P
        Matrix KH = mult(K, H);
        Matrix I_KH = sub(I, KH);
        P = mult(I_KH, P);
    }
};

#endif // KALMAN_CPU_H
