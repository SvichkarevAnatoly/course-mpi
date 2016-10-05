#include <stdio.h>
#include <float.h>
#include <math.h>

const int N = 3;
const double EPS = 0.001;

void printMatrix(double A[][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%f ", A[i][j]);
        }
        printf("\n");
    }
}

void printMatrixAndB(double A[][N], double B[]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%4.2f\t", A[i][j]);
        }
        printf("   \t%4.2f\n", B[i]);
    }
    printf("\n");
}

void initMatrixAndB(double A[][N], double B[]) {
    // 10 1 1
    // 2 10 1
    // 2 2 10
    A[0][0] = 10;
    A[0][1] = 1;
    A[0][2] = 1;

    A[1][0] = 2;
    A[1][1] = 10;
    A[1][2] = 1;

    A[2][0] = 2;
    A[2][1] = 2;
    A[2][2] = 10;

    B[0] = 12;
    B[1] = 13;
    B[2] = 14;
}

void initX(double *X) {
    X[0] = 1.2;
    X[1] = 0;
    X[2] = 0;
}

void expressionVariables(double A[][N], double B[]) {
    for (int i = 0; i < N; ++i) {
        double cur = A[i][i];
        for (int j = 0; j < N; ++j) {
            A[i][j] = -A[i][j] / cur;
        }
        B[i] /= cur;
        A[i][i] = 0;
    }
}

void copyX(double *X, double *B) {
    for (int i = 0; i < N; ++i) {
        X[i] = B[i];
    }
}

void printX(double X[]) {
    for (int i = 0; i < N; ++i) {
        printf("%4.4f\t", X[i]);
    }
    printf("\n");
}

double diffX(double X[], double oldX[]) {
    double maxDiff = -DBL_MAX;
    for (int i = 0; i < N; ++i) {
        if (fabs(X[i] - oldX[i]) > maxDiff) {
            maxDiff = fabs(X[i] - oldX[i]);
        }
    }

    return maxDiff;
}

void zeroX(double X[]) {
    for (int i = 0; i < N; ++i) {
        X[i] = 0;
    }
}

int main() {
    double A[N][N];
    double B[N];

    initMatrixAndB(A, B);
    printMatrixAndB(A, B);

    expressionVariables(A, B);
    printMatrixAndB(A, B);

    double X[N];
    initX(X);
    printX(X);

    double oldX[N];
    do {
        copyX(oldX, X);
        for (int i = 0; i < N; ++i) {
            X[i] = A[i][0] * X[0];
            for (int j = 1; j < N; ++j) {
                X[i] += A[i][j] * X[j];
            }
            X[i] += B[i];
        }
        printX(X);
        printf("diff %4.4f\n", diffX(X, oldX));
    } while (diffX(X, oldX) > EPS);

    return 0;
}
