      module arrays
      implicit none
      integer,allocatable :: listOffset(:,:),lists(:,:),boxes(:,:)
      integer,allocatable :: levelOffset(:),nodes(:,:)
      real *8,allocatable :: centers(:,:),corners(:,:,:)
      end module
