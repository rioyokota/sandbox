program main
  include 'mpif.h'
  character*256 hostname
  integer ierr, rank, size
  call mpi_init(ierr)
  call mpi_comm_size(mpi_comm_world, size, ierr)
  call mpi_comm_rank(mpi_comm_world, rank, ierr)
  call getenv('HOST',hostname)
  do irank=1,size
     call mpi_barrier(mpi_comm_world, ierr)
     if (rank.eq.irank) then
        print*,hostname,' ',rank,' / ',size
     end if
  end do
  call mpi_finalize(ierr)
end program main
