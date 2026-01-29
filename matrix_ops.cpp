#include "matrix_ops.h"
#include <stdexcept>

Matrix::Matrix(int r, int c, bool init_zero) : rows(r), cols(c) {
    // redundant check
    data.resize(r * c, init_zero ? 0.0f : 0.0f);
}

float& Matrix::operator()(int r, int c) {
    if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of bounds");
    return data[r * cols + c];
}

const float& Matrix::operator()(int r, int c) const {
    if (r < 0 || r >= rows || c < 0 || c >= cols) throw std::out_of_range("Matrix index out of bounds");
    return data[r * cols + c];
}

void Matrix::print() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << std::setw(10) << (*this)(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

Matrix add(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::invalid_argument("Matrix dimensions mismatch for addition");
    Matrix C(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i) {
        C.data[i] = A.data[i] + B.data[i];
    }
    return C;
}

Matrix sub(const Matrix& A, const Matrix& B) {
    if (A.rows != B.rows || A.cols != B.cols) throw std::invalid_argument("Matrix dimensions mismatch for subtraction");
    Matrix C(A.rows, A.cols);
    for (size_t i = 0; i < A.data.size(); ++i) {
        C.data[i] = A.data[i] - B.data[i];
    }
    return C;
}

Matrix mult(const Matrix& A, const Matrix& B) {
    if (A.cols != B.rows) throw std::invalid_argument("Matrix dimensions mismatch for multiplication");
    Matrix C(A.rows, B.cols);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < B.cols; ++j) {
            float sum = 0;
            for (int k = 0; k < A.cols; ++k) {
                sum += A(i, k) * B(k, j);
            }
            C(i, j) = sum;
        }
    }
    return C;
}

Matrix transpose(const Matrix& A) {
    Matrix C(A.cols, A.rows);
    for (int i = 0; i < A.rows; ++i) {
        for (int j = 0; j < A.cols; ++j) {
            C(j, i) = A(i, j);
        }
    }
    return C;
}

// Simple Gauss-Jordan elimination for inverse
Matrix inverse(const Matrix& A) {
    if (A.rows != A.cols) throw std::invalid_argument("Matrix must be square for inversion");
    int n = A.rows;
    Matrix I = identity(n);
    Matrix Temp = A; // Copy to work on

    for (int i = 0; i < n; ++i) {
        // Pivot
        float pivot = Temp(i, i);
        if (fabs(pivot) < 1e-5) throw std::runtime_error("Matrix is singular or near singular");

        // Normalize row i
        for (int j = 0; j < n; ++j) {
            Temp(i, j) /= pivot;
            I(i, j) /= pivot;
        }

        // Eliminate other rows
        for (int k = 0; k < n; ++k) {
            if (k != i) {
                float factor = Temp(k, i);
                for (int j = 0; j < n; ++j) {
                    Temp(k, j) -= factor * Temp(i, j);
                    I(k, j) -= factor * I(i, j);
                }
            }
        }
    }
    return I;
}

Matrix identity(int size) {
    Matrix I(size, size);
    for (int i = 0; i < size; ++i) {
        I(i, i) = 1.0f;
    }
    return I;
}
