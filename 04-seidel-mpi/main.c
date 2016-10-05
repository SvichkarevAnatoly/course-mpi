#include <mpi.h>
#include <stdio.h>
#include <malloc.h>

void generate_data(int *buf, int size) {
    int i;
    for (i = 0; i < size; ++i) {
        buf[i] = i;
    }
}

void print_data(int buf[], const int size, int rank) {
    int i;
    for (i = 0; i < size; ++i) {
        printf("r%d %d\n", rank, buf[i]);
    }
    printf("\n");
}

int main(int argc, char *argv[]) {
    int rank, size;

    const int recvsize = 50;
    int recvbuf[recvsize];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int sendsize = size * recvsize;
    int *sendbuf = (int *) malloc(sendsize * sizeof(int));

    if (rank == 0) {
        generate_data(sendbuf, sendsize);
    }

    MPI_Scatter(sendbuf, recvsize, MPI_INT, recvbuf, recvsize, MPI_INT, 0, MPI_COMM_WORLD);

    print_data(recvbuf, recvsize, rank);

    MPI_Finalize();
    return 0;
}