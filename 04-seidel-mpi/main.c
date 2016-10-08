#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

const double EPS = 0.001;

void expressionVariables(double A[], int N) {
    int i, j;
    for (i = 0; i < N; ++i) {
        double cur = A[i * (N + 1) + i];
        for (j = 0; j < N; ++j) {
            A[i * (N + 1) + j] = -A[i * (N + 1) + j] / cur;
        }
        A[i * (N + 1) + N] /= cur;
        A[i * (N + 1) + i] = 0;
    }
}

void copyX(double Xto[], double Xfrom[], int N) {
    int i;
    for (i = 0; i < N; ++i) {
        Xto[i] = Xfrom[i];
    }
}

void prepareScatterA(int sendcountsA[], int displsA[], int size, int N) {
    int i;
    int rem = N % size;
    int sum = 0;
    for (i = 0; i < size; i++) {
        sendcountsA[i] = (N / size) * (N + 1);
        if (rem > 0) {
            sendcountsA[i] += (N + 1);
            rem--;
        }

        displsA[i] = sum;
        sum += sendcountsA[i];
    }
    return;
}

void prepareScatterX(int sendcountsX[], int displsX[], int size, int N) {
    int i;
    int rem = N % size;
    int sum = 0;
    for (i = 0; i < size; i++) {
        sendcountsX[i] = N / size;
        if (rem > 0) {
            sendcountsX[i]++;
            rem--;
        }

        displsX[i] = sum;
        sum += sendcountsX[i];
    }

    return;
}

void initXasB(double X[], double A[], int N) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = A[i * (N + 1) + N];
    }
}

void iteration(double AA[], double X[], int sizeXrank, int N) {
    int i, j;
    for (i = 0; i < sizeXrank / (N + 1); ++i) {
        X[i] = AA[i * (N + 1)] * X[0];
        for (j = 1; j < N; ++j) {
            X[i] += AA[i * (N + 1) + j] * X[j];
        }
        X[i] += AA[i * (N + 1) + N];
    }
}

double localDiffMax(double X[], double oldX[], int sizeXrank, int displsXrank) {
    int i;
    double max = fabs(X[0] - oldX[displsXrank]);
    for (i = 1; i < sizeXrank; ++i) {
        if (fabs(X[i] - oldX[displsXrank + i]) > max) {
            max = fabs(X[i] - oldX[displsXrank + i]);
        }
    }
    return max;
}

void generateW(double w[], int N) {
    int i;
    for (i = 0; i < N; ++i) {
        w[i] = sin(i + 1);
    }
}

void printVector(double w[], int N) {
    printf("Vector:\n");
    int i;
    for (i = 0; i < N; ++i) {
        printf("%4.4f\n", w[i]);
    }
    printf("\n");
}

void printMatrix(double P[], int N) {
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

void printMatrixAF(double AF[], int N) {
    int i, j;
    printf("Matrix AF:\n");
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N + 1; ++j) {
            printf("%5.2f ", AF[i * (N + 1) + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void productW(double K[], double w[], int N) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            K[i * N + j] = w[i] * w[j];
        }
    }
}

double scalarProductW(double w[], int N) {
    int i;
    double sp = 0;
    for (i = 0; i < N; ++i) {
        sp += w[i] * w[i];
    }
    return sp;
}

// P = E - 2 * w*w^T / w^T*w
void generateP(double P[], double w[], int N) {
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
    productW(K, w, N);
    // скалярное произведение
    double ww = scalarProductW(w, N);
    // матрица P
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            P[i * N + j] -= 2 * K[i * N + j] / ww;
        }
    }
}

void matrixProduct(double A[], double B[], double AB[], int N) {
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
void generateL(double A[], int cond, int N) {
    int i, j;
    double minLambda = 1.0 / cond;
    double maxLambda = 1.0;
    double diffLambda = maxLambda - minLambda;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            A[i * N + j] = 0;
            if (i == j) {
                A[i * N + j] = minLambda + (diffLambda * i) / (N - 1);
            }
        }
    }
}

void generateX(double X[], int N) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = 1;
    }
}

void computeF(double A[], double X[], double F[], int N) {
    int i, j;
    for (i = 0; i < N; ++i) {
        F[i] = 0;
        for (j = 0; j < N; ++j) {
            F[i] += A[i * N + j] * X[j];
        }
    }
}

void merge(double A[], double F[], double AF[], int N) {
    int i, j;
    for (i = 0; i < N; ++i) {
        AF[i * (N + 1) + N] = F[i];
        for (j = 0; j < N; ++j) {
            AF[i * (N + 1) + j] = A[i * N + j];
        }
    }
}

void generateAF(double AF[], int cond, int N) {
    double w[N];
    double P[N * N];
    double L[N * N]; // lambda
    double PL[N * N];
    double A[N * N]; // PLP
    double X[N];
    double F[N];

    generateW(w, N);
    generateP(P, w, N);
    generateL(L, cond, N);
    matrixProduct(P, L, PL, N);
    matrixProduct(PL, P, A, N);

    generateX(X, N);

    computeF(A, X, F, N);

    merge(A, F, AF, N);
}

void printAverage(double X[], int N) {
    int i;
    double sum = 0;
    for (i = 0; i < N; ++i) {
        sum += X[i];
    }
    printf("Average Xi: %4.4f\n", sum / N);
    return;
}

int main(int argc, char *argv[]) {
    int N = 0;
    int cond = 0;
    if (argc > 1) {
        N = atoi(argv[1]);
        cond = atoi(argv[2]);
    }

    int k = 0;
    int rank = 0, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // вроде для отключения буферизации
    setvbuf(stdout, NULL, _IONBF, 0);

    // в массиве A хранится матрица A и вектор F
    double A[(N + 1) * N];
    double X[N], oldX[N];

    // инициализация матрицы A, векторов X, B
    if (rank == 0) {
        generateAF(A, cond, N);
        expressionVariables(A, N);
        //printMatrixAF(A, N);
        initXasB(X, A, N);
    }

    // вычисление по сколько отправлять данных матрицы A
    int sendcountsA[size];
    int displsA[size];
    prepareScatterA(sendcountsA, displsA, size, N);
    double AA[sendcountsA[rank]];

    // отправление всем процессам частей A
    MPI_Scatterv(A, sendcountsA, displsA, MPI_DOUBLE,
                 AA, sendcountsA[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // вычисление по сколько обмениваться частями X
    int sendcountsX[size];
    int displsX[size];
    prepareScatterX(sendcountsX, displsX, size, N);

    // рассылка всем начального X
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double localmax;
    double globmax;
    int counter = 0;
    do {
        copyX(oldX, X, N);
        iteration(AA, X, sendcountsA[rank], N);

        // вычисление локального максимума на процессе
        localmax = localDiffMax(X, oldX, sendcountsX[rank], displsX[rank]);

        // с помощью all reduce отправить max из max всем остальным
        MPI_Allreduce(&localmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        MPI_Allgatherv(X, sendcountsX[rank], MPI_DOUBLE,
                       X, sendcountsX, displsX, MPI_DOUBLE, MPI_COMM_WORLD);
        counter++;
    } while (globmax > EPS);

    if (rank == 0) {
        //printVector(X, N);
        printAverage(X, N);
        printf("Number iteration: %d\n", counter);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("End rank %d\n", rank);
    MPI_Finalize();
    return 0;
}