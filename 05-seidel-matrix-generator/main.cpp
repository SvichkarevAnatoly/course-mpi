#include <stdio.h>
#include <cmath>

const int N = 3;
const double COND = 1000;

void generateW(double w[]) {
    int i;
    for (i = 0; i < N; ++i) {
        w[i] = sin(i + 1);
    }
}

void printVector(double *w) {
    printf("Vector:\n");
    int i;
    for (i = 0; i < N; ++i) {
        printf("%4.4f\n", w[i]);
    }
    printf("\n");
}

void productW(double K[], double w[]) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            K[i * N + j] = w[i] * w[j];
        }
    }
}

double scalarProductW(double *w) {
    int i;
    double sp = 0;
    for (i = 0; i < N; ++i) {
        sp += w[i] * w[i];
    }
    return sp;
}

void printMatrix(double *P) {
    int i, j;
    printf("Matrix:\n");
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            printf("%4.4f ", P[i * N + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// P = E - 2 * w*w^T / w^T*w
void generateP(double P[], double w[]) {
    int i, j;
    // единичная матрица
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            P[i * N + j] = 0;
            if (i == j) {
                P[i * N + j] = 1;
            }
        }
    }

    double K[N * N];
    // произведение вектора на вектор, дающее матрицу
    productW(K, w);
    // скалярное произведение
    double ww = scalarProductW(w);
    // матрица P
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            P[i * N + j] -= 2 * K[i * N + j] / ww;
        }
    }
}

void matrixProduct(double A[], double B[], double AB[]) {
    int i, j, k;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            AB[i * N + j] = 0;
            for (k = 0; k < N; ++k) {
                AB[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

// с числом обусловленности через собственные значения
void generateL(double *A) {
    int i, j;
    double min_lambda = 1.0 / COND;
    double max_lambda = 1.0;
    double diff_lambda = max_lambda - min_lambda;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            A[i * N + j] = 0;
            if (i == j) {
                A[i * N + j] = min_lambda + (diff_lambda * i) / (N - 1);
            }
        }
    }
}

void generateX(double X[]) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = 1;
    }
}

void computeF(double A[], double X[], double F[]) {
    int i, j;
    for (i = 0; i < N; ++i) {
        F[i] = 0;
        for (j = 0; j < N; ++j) {
            F[i] += A[i * N + j] * X[j];
        }
    }
}

int main() {
    double w[N];
    double P[N * N];
    double L[N * N]; // lambda
    double PL[N * N];
    double A[N * N]; // PLP
    double X[N];
    double F[N];

    generateW(w);
    generateP(P, w);
    generateL(L);
    matrixProduct(P, L, PL);
    matrixProduct(PL, P, A);
    printMatrix(A);

    generateX(X);
    printVector(X);

    computeF(A, X, F);
    printVector(F);

    printf("End\n");
    return 0;
}