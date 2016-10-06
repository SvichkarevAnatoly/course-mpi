#include <mpi.h>
#include <stdio.h>
#include <malloc.h>
#include <zconf.h>
#include <math.h>

#pragma clang diagnostic push
#pragma ide diagnostic ignored "OCDFAInspection"
const int N = 3;
const double EPS = 0.001;

void printMatrixAndB(double A[]) {
    int i, j;
    for (i = 0; i < N; ++i) {
        for (j = 0; j < N; ++j) {
            printf("%4.2f\t", A[i * (N + 1) + j]);
        }
        printf("   \t%4.2f\n", A[i * (N + 1) + N]);
    }
    printf("\n");
    fflush(stdout);
}

void initAndF(double *A) {
    // A
    // 0.5082 0.0564 -0.0675
    // 0.0564 0.0302 -0.1554
    // -0.0675 -0.1554 0.9631
    // F
    // 0.4971 -0.0688 0.7401
    A[0] = 0.5082;
    A[1] = 0.0564;
    A[2] = -0.0675;
    A[3] = 0.4971;

    A[4] = 0.0564;
    A[5] = 0.0302;
    A[6] = -0.1554;
    A[7] = -0.0688;

    A[8] = -0.0675;
    A[9] = -0.1554;
    A[10] = 0.9631;
    A[11] = 0.7401;
}

void printDoubleArray(double *arr, int size, int rank) {
    int i;
    for (i = 0; i < size; ++i) {
        printf("r%d %4.2f\t", rank, arr[i]);
    }
    printf("\n");
    fflush(stdout);
}

void expressionVariables(double *A) {
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

void copyX(double *Xto, double *Xfrom) {
    int i;
    for (i = 0; i < N; ++i) {
        Xto[i] = Xfrom[i];
    }
}

void zeroX(double X[]) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = 0;
    }
}

void prepare_scatterA(int *sendcountsA, int *displsA, int size) {
    int i;
    int rem = N % size;   // elements remaining after division among processes
    int sum = 0;          // Sum of counts. Used to calculate displacements
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

void prepare_scatterX(int *sendcountsX, int *displsX, int size) {
    int i;
    int rem = N % size;   // elements remaining after division among processes
    int sum = 0;          // Sum of counts. Used to calculate displacements
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

void initXasB(double *X, double *A) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = A[i * (N + 1) + N];
    }
}

void iteration(double *AA, double *X, int sizeXrank) {
    int i, j;
    double oldX[N];
    zeroX(oldX);
    copyX(oldX, X);
    for (i = 0; i < sizeXrank / (N + 1); ++i) {
        X[i] = AA[i * (N + 1)] * X[0];
        for (j = 1; j < N; ++j) {
            X[i] += AA[i * (N + 1) + j] * X[j];
        }
        X[i] += AA[i * (N + 1) + N];
    }
}

double local_max(double X[], double oldX[], int sizeXrank, int displsXrank) {
    int i;
    double max = fabs(X[0] - oldX[displsXrank]);
    for (i = 1; i < sizeXrank; ++i) {
        if (fabs(X[i] - oldX[displsXrank + i]) > max) {
            max = fabs(X[i] - oldX[displsXrank + i]);
        }
    }
    return max;
}

int main(int argc, char *argv[]) {
    int i = 0, j = 0, k = 0;
    int rank, size;

    double A[(N + 1) * N];
    double X[N], oldX[N];

    // вроде для отключения буферизации
    setvbuf(stdout, NULL, _IONBF, 0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // for debugger
    /*char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    //while (rank == 1 && 0 == k)
    while (rank == 0 && 0 == k)
        sleep(5);*/
    // end for debugger

    // инициализация матрицы A, векторов X, B
    if (rank == 0) {
        initAndF(A);
        expressionVariables(A);
        initXasB(X, A);
    }

    // вычисление по сколько отправлять данных матрицы A
    int *sendcountsA = malloc(sizeof(int) * size);
    int *displsA = malloc(sizeof(int) * size);
    prepare_scatterA(sendcountsA, displsA, size);
    double *AA = malloc(sizeof(double) * sendcountsA[rank]);

    // отправление всем процессам частей A
    MPI_Scatterv(A, sendcountsA, displsA, MPI_DOUBLE,
                 AA, sendcountsA[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // вычисление по сколько обмениваться частями X
    int *sendcountsX = malloc(sizeof(int) * size);
    int *displsX = malloc(sizeof(int) * size);
    prepare_scatterX(sendcountsX, displsX, size);

    // рассылка всем начального X
    MPI_Bcast(X, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double localmax;
    double globmax;
    do {
        copyX(oldX, X);
        iteration(AA, X, sendcountsA[rank]);

        // вычисление локального максимума на процессе
        localmax = local_max(X, oldX, sendcountsX[rank], displsX[rank]);

        // с помощью all reduce отправить max из max всем остальным
        MPI_Allreduce(&localmax, &globmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        // printf("globmax = %4.4f\n", globmax);

        MPI_Allgatherv(X, sendcountsX[rank], MPI_DOUBLE,
                       X, sendcountsX, displsX, MPI_DOUBLE, MPI_COMM_WORLD);
    } while (globmax > EPS);

    if (rank == 0) {
        printDoubleArray(X, N, rank);
    }

    printf("End\n");
    MPI_Finalize();
    return 0;
}

#pragma clang diagnostic pop

/* 1) Почему в MPI_Scatterv выходной и входной буфер не могут совпадать?
 * 2) Как перевести стандарт компилятора, чтобы в for можно было объявлять переменные?
 * */