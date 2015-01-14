      subroutine fmm(iprec,wavek,numBodies,Xj,qj,pi,Fi)
      use constants, only : P,maxLevel
      use arrays, only : levelOffset,cells
      use omp_lib, only : omp_get_wtime
      implicit none
      integer iprec,ncrit,numCells,numLevels,i,icell,numBodies
      integer level,sumTerms
      integer, allocatable :: iaddr(:)
      integer, allocatable :: permutation(:)
      real *8 tolerance,R0,tic/0.0d0/,toc/0.0d0/
      real *8 X0(3)
      real *8 Xj(3,numBodies)
      real *8 Xjd(3,numBodies)
      real *8 scale(0:maxLevel)
      real *8, allocatable :: Multipole(:)
      real *8, allocatable :: Local(:)
      complex *16 wavek
      complex *16 qj(numBodies)
      complex *16 qjd(numBodies)
      complex *16 pi(numBodies)
      complex *16 Fi(3,numBodies)
      complex *16 pid(numBodies)
      complex *16 Fid(3,numBodies)
c     set criterion for cell subdivision (number of sources per cell)
      if (iprec.eq.-2) ncrit=40
      if (iprec.eq.-1) ncrit=50
      if (iprec.eq.0) ncrit=80
      if (iprec.eq.1) ncrit=160
      if (iprec.eq.2) ncrit=400
      if (iprec.eq.3) ncrit=800
      if (iprec.eq.4) ncrit=1200
      if (iprec.eq.5) ncrit=1400
      if (iprec.eq.6) ncrit=numBodies
c     create oct-tree data structure
      allocate (permutation(numBodies))
      allocate (levelOffset(maxLevel))
c$    tic=omp_get_wtime()
c$    tic=omp_get_wtime()
      call getBounds(Xj,numBodies,X0,R0)
      call getTolerance(2*R0,wavek,P,tolerance)
      call buildTree(Xj,numBodies,ncrit,
     1     numCells,permutation,numLevels,X0,R0)
      allocate(iaddr(numCells))
      do i=0,numLevels
         scale(i)=(R0/2.0**(i-1))*wavek
         if (scale(i).ge.1) scale(i)=1.0
      enddo
      do i=1,numBodies
         Xjd(1,i)=Xj(1,permutation(i))
         Xjd(2,i)=Xj(2,permutation(i))
         Xjd(3,i)=Xj(3,permutation(i))
         qjd(i)=qj(permutation(i))
      enddo
      sumTerms=1
      do icell=1,numCells
         level=cells(1,icell)
         iaddr(icell)=sumTerms
         sumTerms=sumTerms+(P+1)*(2*P+1)*2
      enddo
      allocate(Multipole(sumTerms))
      allocate(Local(sumTerms))
c$    toc=omp_get_wtime()
      print*,'Tree   =',toc-tic
      call evaluate(wavek,numBodies,Xjd,
     1     qjd,pid,Fid,tolerance,iaddr,Multipole,Local,
     1     numCells,numLevels,scale,R0)
      do i=1,numBodies
         pi(permutation(i))=pid(i)
         Fi(1,permutation(i))=Fid(1,i)
         Fi(2,permutation(i))=Fid(2,i)
         Fi(3,permutation(i))=Fid(3,i)
      enddo
      return
      end

      subroutine evaluate(wavek,numBodies,Xj,qj,pi,Fi,
     1     tolerance,iaddr,Multipole,Local,
     1     numCells,numLevels,scale,R0)
      use constants, only : P,Pmax,maxLevel
      use arrays, only : levelOffset,cells,centers
      use omp_lib, only : omp_get_wtime
      implicit none
      integer i,numBodies,numLevels,icell,jcell
      integer nquad,level,ilist,nlist,ntrunc
      integer ii,jj,kk
      integer numCells,ibegin,isize
      integer iaddr(*),list(189)
      integer itable(-3:3,-3:3,-3:3)
      real *8 tolerance,radius,diameter,R0,tic/0.0d0/,toc/0.0d0/
      real *8 Xj(3,*)
      real *8 Multipole(*),Local(*),xquad(2*Pmax),wquad(2*Pmax)
      real *8 scale(0:maxLevel)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek
      complex *16 pi(1)
      complex *16 Fi(3,*)
      complex *16 qj(1)
      do i=1,numBodies
         pi(i)=0
         Fi(1,i)=0
         Fi(2,i)=0
         Fi(3,i)=0
      enddo
      call getAnm(Pmax,Anm1,Anm2)
      do icell=1,numCells
         level=cells(1,icell)
         call initCoefs(Multipole(iaddr(icell)),P)
         call initCoefs(Local(iaddr(icell)),P)
      enddo

c     ... step 1: P2M
c$    tic=omp_get_wtime()
      do level=2,numLevels
c$omp parallel do default(shared)
c$omp$private(icell,ibegin,isize)
         do icell=levelOffset(level+1),levelOffset(level+2)-1
            if (cells(9,icell).eq.0) cycle
            if (cells(7,icell).eq.0) then
               ibegin=cells(8,icell)
               isize=cells(9,icell)
               call P2M(wavek,scale(level),Xj(1,ibegin),qj(ibegin),
     1              isize,centers(1,icell),
     1              Multipole(iaddr(icell)),Anm1,Anm2,Pmax)
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'P2M    =',toc-tic

c     ... step 2, M2M
c$    tic=omp_get_wtime()
      do level=numLevels,3,-1
         radius=R0/2.0d0**(level-1)*sqrt(3.0)            
         nquad=P*2
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,jcell,ilist)
         do icell=levelOffset(level),levelOffset(level+1)-1
            if (cells(9,icell).eq.0) cycle
            if (cells(7,icell).ne.0) then
               do ilist=1,cells(7,icell)
                  jcell=cells(6,icell)+ilist-1
                  call M2M(wavek,scale(level),centers(1,jcell),
     1                 Multipole(iaddr(jcell)),
     1                 scale(level-1),centers(1,icell),
     1                 Multipole(iaddr(icell)),
     1                 radius,xquad,wquad,nquad,Anm1,Anm2,Pmax)
               enddo
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'M2M    =',toc-tic

c     ... step 3, M2L
c$    tic=omp_get_wtime()
      do level=2,numLevels
         diameter=R0/2.0d0**(level-1)
         radius=diameter*sqrt(3.0)*0.5
         call getNumTermsList(diameter,wavek,tolerance,itable)
         nquad=P
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,jcell,list,ilist,nlist)
c$omp$private(ntrunc,ii,jj,kk)
c$omp$schedule(dynamic)
         do icell=levelOffset(level+1),levelOffset(level+2)-1
            call getList(2,icell,list,nlist)
            do ilist=1,nlist
               jcell=list(ilist)
               if (cells(9,jcell).eq.0) cycle
               ii=cells(2,jcell)-cells(2,icell)
               jj=cells(3,jcell)-cells(3,icell)
               kk=cells(4,jcell)-cells(4,icell)
               ntrunc=itable(ii,jj,kk)
               call M2L(wavek,scale(level),
     1              centers(1,jcell),Multipole(iaddr(jcell)),
     1              scale(level),centers(1,icell),Local(iaddr(icell)),
     1              ntrunc,radius,xquad,wquad,nquad,Anm1,Anm2,Pmax)
            enddo
         enddo
      enddo
c$    toc=omp_get_wtime()
      print*,'M2L    =',toc-tic

c     ... step 4, L2L
c$    tic=omp_get_wtime()
      do level=3,numLevels
         radius=R0/2.0d0**(level-1)*sqrt(3.0)
         nquad=P
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,jcell,ilist)
         do icell=levelOffset(level),levelOffset(level+1)-1
            if (cells(7,icell).ne.0) then
               do ilist=1,cells(7,icell)
                  jcell=cells(6,icell)+ilist-1
                  call L2L(wavek,scale(level-1),centers(1,icell),
     1                 Local(iaddr(icell)),
     1                 scale(level),centers(1,jcell),
     1                 Local(iaddr(jcell)),
     1                 radius,xquad,wquad,nquad,
     1                 Anm1,Anm2,Pmax)
               enddo
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'L2L    =',toc-tic

c     ... step 5: L2P
c$    tic=omp_get_wtime()
      do level=2,numLevels
c$omp parallel do default(shared)
c$omp$private(icell,ibegin,isize)
         do icell=levelOffset(level+1),levelOffset(level+2)-1
            if (cells(9,icell).eq.0) cycle
            if (cells(7,icell).eq.0) then
               ibegin=cells(8,icell)
               isize=cells(9,icell)
               call L2P(wavek,scale(level),centers(1,icell),
     1              Local(iaddr(icell)),
     1              Xj(1,ibegin),isize,pi(ibegin),Fi(1,ibegin),
     1              Anm1,Anm2,Pmax)
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'L2P    =',toc-tic

c     ... step 6: P2P
c$    tic=omp_get_wtime()
c$omp parallel do default(shared)
c$omp$private(icell,jcell,list,ilist,nlist)
c$omp$schedule(dynamic)
      do icell=1,numCells
         if (cells(7,icell).eq.0) then
            call P2P(cells(1,icell),pi,Fi,
     1           cells(1,icell),Xj,qj,wavek)
            call getList(1,icell,list,nlist)
            do ilist=1,nlist
               jcell=list(ilist)
               if (cells(9,jcell).eq.0) cycle
               call P2P(cells(1,icell),pi,Fi,
     1              cells(1,jcell),Xj,qj,wavek)
            enddo
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'P2P    =',toc-tic
      return
      end
