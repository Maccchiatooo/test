#include "mpi.h"
#include <Kokkos_Core.hpp>
#include <unistd.h>

using namespace Kokkos;
using buffer_cuda = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaSpace>;
using buffer_uvm = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace>;
using buffer_pin = Kokkos::View<double *, Kokkos::LayoutLeft, Kokkos::CudaHostPinnedSpace>;
int main(int argc, char *argv[])
{

    double start, end;
    MPI_Init(&argc, &argv);
    Kokkos::initialize(argc, argv);
    {

        int nranks;
        int me;
        int size=1e8;
        int neighbor;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm_rank(MPI_COMM_WORLD,  &me);
        MPI_Request request,request1;

        buffer_cuda send=buffer_cuda("send", size);
        Kokkos::deep_copy(send, 1.0);
        buffer_cuda recv = buffer_cuda("recv", size);

        buffer_uvm uvm_send = buffer_uvm("uvmsend", size);
        buffer_uvm uvm_recv = buffer_uvm("uvmrecv", size);

        buffer_pin pin_send = buffer_pin("pinsend", size);
        buffer_pin pin_recv = buffer_pin("pinrecv", size);


        MPI_Datatype column_type;
        MPI_Type_vector(1, size, size/10, MPI_DOUBLE, &column_type);
        MPI_Type_commit(&column_type);

        if(me==0)
            neighbor = 1;
        if(me==1)
            neighbor = 0;

        start = MPI_Wtime();

        Kokkos::deep_copy(pin_send, send);
        MPI_Isend(pin_send.data(), 1, column_type, neighbor, 0, MPI_COMM_WORLD, &request);
        MPI_Irecv(pin_recv.data(), 1, column_type, neighbor, 0,MPI_COMM_WORLD, &request1);
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        MPI_Wait(&request1, MPI_STATUS_IGNORE);
        Kokkos::deep_copy(recv, pin_recv);

        end = MPI_Wtime();
        if(me==0)
            printf("time=%f\n", end - start);

    }
    Kokkos::finalize();
    MPI_Finalize();
}
