#ifndef MATRIX_OPS_H
#define MATRIX_OPS_H

#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>



struct Matrix {
    int rows;
    int cols;
    std::vector<float> data;

    Matrix(int r, int c, bool init_zero = true);
    // add boundary check
    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;
    void print() const;
};

// Matrix Operations
Matrix add(const Matrix& A, const Matrix& B);
Matrix sub(const Matrix& A, const Matrix& B);
Matrix mult(const Matrix& A, const Matrix& B);
Matrix transpose(const Matrix& A);
Matrix inverse(const Matrix& A);
Matrix identity(int size);

#endif // MATRIX_OPS_H
