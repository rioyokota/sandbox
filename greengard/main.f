      program main
      use omp_lib, only : omp_get_wtime
      implicit none
      integer numBodies,numTarget,i
      integer icell(10),jcell(10)
      real*8 pdiff,pnorm,fdiff,fnorm,tic/0.0/,toc/0.0/
      real*8 Xj(3,1000000)
      complex*16 qj(1000000)
      complex*16 pi(1000000)
      complex*16 Fi(3,1000000)
      complex*16 pi2(1000000)
      complex*16 Fi2(3,1000000)
      complex*16 wavek/(10.0d0,1.0d0)/,imag/(0.0d0,1.0d0)/
      numBodies=100000
      call random_number(Xj)
      print*,'N      =',numBodies
      do i=1,numBodies
         qj(i)=Xj(1,i)+imag*Xj(2,i)
      enddo
c$    tic=omp_get_wtime()
      call fmm(wavek,numBodies,Xj,qj,pi,Fi)
c$    toc=omp_get_wtime()
      print*,'FMM    =',toc-tic
      numTarget=min(numBodies,100)
      do i=1,numTarget
         pi2(i)=0
         Fi2(1,i)=0
         Fi2(2,i)=0
         Fi2(3,i)=0
      enddo
      icell(8)=1
      icell(9)=numTarget
      jcell(8)=1
      jcell(9)=numBodies
c$    tic=omp_get_wtime()
      call P2P(icell,pi2,Fi2,jcell,Xj,qj,wavek)
c$    toc=omp_get_wtime()
      print*,'Direct =',toc-tic
      pdiff=0
      pnorm=0
      fdiff=0
      fnorm=0
      do i=1,numTarget
         pdiff=pdiff+abs(pi(i)-pi2(i))**2
         pnorm=pnorm+abs(pi2(i))**2
         fdiff=fdiff+abs(Fi(1,i)-Fi2(1,i))**2
     1        +abs(Fi(2,i)-Fi2(2,i))**2
     1        +abs(Fi(3,i)-Fi2(3,i))**2
         fnorm=fnorm+abs(Fi2(1,i))**2
     1        +abs(Fi2(2,i))**2
     1        +abs(Fi2(3,i))**2
      enddo
      print*,'Err pot=',sqrt(pdiff/pnorm)
      print*,'Err acc=',sqrt(fdiff/fnorm)
      stop
      end
