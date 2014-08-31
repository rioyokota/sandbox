      subroutine rotate(theta,ntermsj,Mnm,ntermsi,Mrot)
      implicit none
      integer ntermsj,ntermsi
      real *8 theta
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsi,-ntermsi:ntermsi)
      call rotviarecur3f90(theta,ntermsj,Mnm,ntermsj,Mrot,ntermsi)
      return
      end
