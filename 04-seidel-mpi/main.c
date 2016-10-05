#include <mpi.h>
#include <stdio.h>
#include <malloc.h>
#include <zconf.h>
#include <float.h>
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

void initMatrixAndB(double A[]) {
    // 10 1 1  12
    // 2 10 1  13
    // 2 2 10  14
    A[0] = 10;
    A[1] = 1;
    A[2] = 1;
    A[3] = 12;

    A[4] = 2;
    A[5] = 10;
    A[6] = 1;
    A[7] = 13;

    A[8] = 2;
    A[9] = 2;
    A[10] = 10;
    A[11] = 14;
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

double diffX(double X[], double oldX[]) {
    int i;
    double maxDiff = -DBL_MAX;
    for (i = 0; i < N; ++i) {
        if (fabs(X[i] - oldX[i]) > maxDiff) {
            maxDiff = fabs(X[i] - oldX[i]);
        }
    }

    return maxDiff;
}

void zeroX(double X[]) {
    int i;
    for (i = 0; i < N; ++i) {
        X[i] = 0;
    }
}

int main(int argc, char *argv[]) {
    int i = 0, j = 0, k = 0;
    int rank, size;

    double A[(N + 1) * N];
    double *AA = 0;

    int *sendcounts;    // array describing how many elements to send to each process
    int *displs;        // array describing the displacements where each segment begins

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rem = N % size;   // elements remaining after division among processes
    int sum = 0;          // Sum of counts. Used to calculate displacements

    // for debugger
    /*char hostname[256];
    gethostname(hostname, sizeof(hostname));
    printf("PID %d on %s ready for attach\n", getpid(), hostname);
    fflush(stdout);
    //while (rank == 1 && 0 == k)
    while (rank == 0 && 0 == k)
        sleep(5);*/
    // end for debugger

    if (rank == 0) {
        initMatrixAndB(A);
        //printMatrixAndB(A);
        expressionVariables(A);
        //printMatrixAndB(A);
    }

    sendcounts = malloc(sizeof(int) * size);
    displs = malloc(sizeof(int) * size);

    // calculate send counts and displacements
    for (i = 0; i < size; i++) {
        sendcounts[i] = (N / size) * (N + 1);
        if (rem > 0) {
            sendcounts[i] += (N + 1);
            rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
        if (rank == i) {
            AA = malloc(sizeof(double) * sendcounts[i]);
        }
    }

    // print calculated send counts and displacements for each process
    /*if (0 == rank) {
        for (i = 0; i < size; i++) {
            printf("sendcounts[%d] = %d\tdispls[%d] = %d\n",
                   i, sendcounts[i], i, displs[i]);
        }
    }*/

    // divide the data among processes as described by sendcounts and displs
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 AA, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        //printDoubleArray(AA, sendcounts[0], rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 1) {
        //printDoubleArray(AA, sendcounts[1], rank);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    double X[N];
    double oldX[N];
    zeroX(X);
    zeroX(oldX);

    X[0] = 1.2;
    X[1] = 0;
    X[2] = 0;

    copyX(oldX, X);
    for (i = 0; i < sendcounts[rank] / (N + 1); ++i) {
        X[i] = AA[i * (N + 1)] * X[0];
        for (j = 1; j < N; ++j) {
            X[i] += AA[i * (N + 1) + j] * X[j];
        }
        X[i] += AA[i * (N + 1) + N];
    }
    //printDoubleArray(X, N, rank);
    //printf("diff %4.4f\n", diffX(X, oldX));

    double XX[N]; // только для 0 процесса
    int sendcountsX[2] = {2, 1};
    int displsX[2] = {0, 2};

    if (rank == 1) {
        //printf("?????\n");
        //printDoubleArray(X, N, rank);
    }

    MPI_Gather(X, 2, MPI_DOUBLE,
               XX, sendcountsX[rank], MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("========\n");
        printDoubleArray(XX, N, rank);
        printf("Diff %4.4f\n", diffX(X, XX));
    }

    zeroX(X);

    sum = 0;
    rem = N % size;
    for (i = 0; i < size; i++) {
        sendcounts[i] = N / size;
        if (rem > 0) {
            sendcounts[i]++;
            rem--;
        }

        displs[i] = sum;
        sum += sendcounts[i];
    }
    if (0 == rank) {
        for (i = 0; i < size; i++) {
            printf("sendcounts[%d] = %d\tdispls[%d] = %d\n",
                   i, sendcounts[i], i, displs[i]);
        }
    }
    MPI_Scatterv(XX, sendcounts, displs, MPI_DOUBLE,
                 X, sendcounts[rank], MPI_DOUBLE,
                 0, MPI_COMM_WORLD);
    printDoubleArray(X, N, rank);

    fflush(stdout);

    MPI_Finalize();
    return 0;
}

#pragma clang diagnostic pop

/*
 * 1) Почему вывод каждый раз в разном месте прерывается
 * 2) Как дебажить?
 * 3) Как использовать gatherv если куски для приёма разные?
 */