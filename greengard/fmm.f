      subroutine fmm(iprec,wavek,numBodies,Xj,qj,pi,Fi)
      use arrays, only : levelOffset,cells
      use omp_lib, only : omp_get_wtime
      implicit none
      integer iprec,ncrit,numCells,nlev,i,nmax,icell,numBodies
      integer level,sumTerms
      integer nterms(0:200)
      integer, allocatable :: iaddr(:)
      integer, allocatable :: permutation(:)
      real *8 epsfmm,R0,tic/0.0d0/,toc/0.0d0/
      real *8 X0(3)
      real *8 Xj(3,numBodies)
      real *8 Xjd(3,numBodies)
      real *8 bsize(0:200)
      real *8 scale(0:200)
      real *8, allocatable :: Multipole(:)
      real *8, allocatable :: Local(:)
      complex *16 wavek
      complex *16 qj(numBodies)
      complex *16 qjd(2*numBodies)
      complex *16 pi(numBodies)
      complex *16 Fi(3,numBodies)
      complex *16 pid(numBodies)
      complex *16 Fid(3,numBodies)
c     set fmm tolerance based on iprec flag.
      if (iprec.eq.-2) epsfmm=.5d-0
      if (iprec.eq.-1) epsfmm=.5d-1
      if (iprec.eq.0) epsfmm=.5d-2
      if (iprec.eq.1) epsfmm=.5d-3
      if (iprec.eq.2) epsfmm=.5d-6
      if (iprec.eq.3) epsfmm=.5d-9
      if (iprec.eq.4) epsfmm=.5d-12
      if (iprec.eq.5) epsfmm=.5d-15
      if (iprec.eq.6) epsfmm=0
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
      allocate (levelOffset(200))
c$    tic=omp_get_wtime()
c$    tic=omp_get_wtime()
      call getBounds(Xj,numBodies,X0,R0)
      call buildTree(Xj,numBodies,ncrit,
     1     numCells,permutation,nlev,X0,R0)
      allocate(iaddr(numCells))
      do i=0,nlev
         scale(i)=(R0/2.0**(i-1))*wavek
         if (scale(i).ge.1) scale(i)=1.0
      enddo
      nmax=0
      do i=0,nlev
         bsize(i)=R0/2.0d0**(i-1)
         call getNumTerms(1,1.5d0,bsize(i),wavek,epsfmm,nterms(i))
         if (nterms(i).gt.nmax.and.i.ge.2) nmax=nterms(i)
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
         sumTerms=sumTerms+(nterms(level)+1)*(2*nterms(level)+1)*2
      enddo
      allocate(Multipole(sumTerms))
      allocate(Local(sumTerms))
c$    toc=omp_get_wtime()
      print*,'Tree   =',toc-tic
      call evaluate(wavek,numBodies,Xjd,
     1     qjd,pid,Fid,epsfmm,iaddr,Multipole,Local,
     1     numCells,nlev,scale,bsize,nterms)
      do i=1,numBodies
         pi(permutation(i))=pid(i)
         Fi(1,permutation(i))=Fid(1,i)
         Fi(2,permutation(i))=Fid(2,i)
         Fi(3,permutation(i))=Fid(3,i)
      enddo
      return
      end

      subroutine evaluate(wavek,
     1     numBodies,Xj,qj,pot,fld,
     1     epsfmm,iaddr,Multipole,Local,
     1     numCells,nlev,scale,bsize,nterms)
      use arrays, only : listOffset,lists,levelOffset,cells,centers
      use omp_lib, only : omp_get_wtime
      implicit none
      integer Pmax,i,numBodies,itype,nlev,icell,ilev
      integer nquad,level,level0,level1,ilist,nbessel
      integer jcell,nlist,nterms_trunc,ii,jj,kk
      integer numCells,ibegin,isize
      integer iaddr(numCells),nterms(0:200),list(10000)
      integer itable(-3:3,-3:3,-3:3)
      integer nterms_eval(4,0:200)
      real *8 epsfmm,radius,tic/0.0d0/,toc/0.0d0/
      real *8 Xj(3,1)
      real *8 Multipole(1),Local(1),xquad(10000),wquad(10000)
      real *8 scale(0:200),bsize(0:200)
      real *8 Anm1(0:200,0:200)
      real *8 Anm2(0:200,0:200)
      complex *16 wavek
      complex *16 pot(1)
      complex *16 fld(3,1)
      complex *16 qj(1)
      do i=1,numBodies
         pot(i)=0
         fld(1,i)=0
         fld(2,i)=0
         fld(3,i)=0
      enddo
      Pmax=200
      call getAnm(Pmax,Anm1,Anm2)
      do i=0,nlev
         do itype=1,4
            call getNumTerms(itype,1.5d0,bsize(i),wavek,epsfmm,
     1           nterms_eval(itype,i))
         enddo
      enddo
      do icell=1,numCells
         level=cells(1,icell)
         call initCoefs(Multipole(iaddr(icell)),nterms(level))
         call initCoefs(Local(iaddr(icell)),nterms(level))
      enddo

c     ... step 1: P2M
c$    tic=omp_get_wtime()
      do ilev=3,nlev+1
c$omp parallel do default(shared)
c$omp$private(icell,level,ibegin,isize)
         do icell=levelOffset(ilev),levelOffset(ilev+1)-1
            level=cells(1,icell)
            nbessel=nterms(level)+1000
            if (cells(9,icell).eq.0) cycle
            if (cells(7,icell).eq.0) then
               ibegin=cells(8,icell)
               isize=cells(9,icell)
               call P2M(wavek,scale(level),
     1              Xj(1,ibegin),qj(ibegin),isize,
     1              centers(1,icell),nterms(level),nterms_eval(1,level),
     1              nbessel,Multipole(iaddr(icell)),Anm1,Anm2,Pmax)
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'P2M    =',toc-tic

c     ... step 2, M2M
c$    tic=omp_get_wtime()
      do ilev=nlev,3,-1
         nquad=nterms(ilev-1)*2.5
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,level0)
c$omp$private(level,radius)
c$omp$private(i,jcell,level1)
         do icell=levelOffset(ilev),levelOffset(ilev+1)-1
            radius=bsize(ilev)*sqrt(3.0)
            if (cells(9,icell).eq.0) cycle
            if (cells(7,icell).ne.0) then
               level0=cells(1,icell)
               if (level0.ge.2) then
                  do i=1,cells(7,icell)
                     jcell=cells(6,icell)+i-1
                     level1=cells(1,jcell)
                     call M2M(wavek,scale(level1),centers(1,jcell),
     1                    Multipole(iaddr(jcell)),nterms(level1),
     1                    scale(level0),centers(1,icell),
     1                    Multipole(iaddr(icell)),
     1                    nterms(level0),
     1                    radius,xquad,wquad,nquad,Anm1,Anm2,Pmax)
                  enddo
               endif
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'M2M    =',toc-tic

c     ... step 3, M2L
c$    tic=omp_get_wtime()
      do ilev=3,nlev+1
         call getNumTermsList(bsize(ilev-1),wavek,epsfmm,itable)
         nquad=nterms(ilev-1)*1.2
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,level0,list,ilist,nlist)
c$omp$private(jcell,level1,radius)
c$omp$private(nterms_trunc,ii,jj,kk)
c$omp$schedule(dynamic)
         do icell=levelOffset(ilev),levelOffset(ilev+1)-1
            radius=bsize(ilev-1)*sqrt(3.0)*0.5
            level0=cells(1,icell)
            if (level0.ge.2) then
               call getList(2,icell,list,nlist)
               do ilist=1,nlist
                  jcell=list(ilist)
                  if (cells(9,jcell).eq.0) cycle
                  level1=cells(1,jcell)
                  ii=cells(2,jcell)-cells(2,icell)
                  jj=cells(3,jcell)-cells(3,icell)
                  kk=cells(4,jcell)-cells(4,icell)
                  nterms_trunc=itable(ii,jj,kk)
                  nterms_trunc=min(nterms(level0),nterms_trunc)
                  nterms_trunc=min(nterms(level1),nterms_trunc)
                  nbessel=nterms_trunc+1000
                  call M2L(wavek,
     1                 scale(level1),
     1                 centers(1,jcell),Multipole(iaddr(jcell)),
     1                 nterms(level1),scale(level0),
     1                 centers(1,icell),Local(iaddr(icell)),
     1                 nterms(level0),nterms_trunc,
     1                 radius,xquad,wquad,nquad,nbessel,
     1                 Anm1,Anm2,Pmax)
               enddo
            endif
         enddo
      enddo
c$    toc=omp_get_wtime()
      print*,'M2L    =',toc-tic

c     ... step 4, L2L
c$    tic=omp_get_wtime()
      do ilev=3,nlev
         nquad=nterms(ilev-1)*2
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(icell,level0)
c$omp$private(level,radius)
c$omp$private(i,jcell,level1)
         do icell=levelOffset(ilev),levelOffset(ilev+1)-1
            radius=bsize(ilev)*sqrt(3.0)
            if (cells(7,icell).ne.0) then
               level0=cells(1,icell)
               if (level0.ge.2) then
                  do i=1,cells(7,icell)
                     jcell=cells(6,icell)+i-1
                     if (jcell.eq.0) cycle
                     level1=cells(1,jcell)
                     nbessel=nquad+1000
                     call L2L(wavek,scale(level0),centers(1,icell),
     1                    Local(iaddr(icell)),nterms(level0),
     1                    scale(level1),centers(1,jcell),
     1                    Local(iaddr(jcell)),nterms(level1),
     1                    radius,xquad,wquad,nquad,nbessel,
     1                    Anm1,Anm2,Pmax)
                  enddo
               endif
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'L2L    =',toc-tic

c     ... step 5: L2P
c$    tic=omp_get_wtime()
c$omp parallel do default(shared)
c$omp$private(icell,level,ibegin,isize)
      do icell=1,numCells
         if (cells(7,icell).eq.0) then
            level=cells(1,icell)
            nbessel=nterms(level)+1000
            if (level.ge.2) then
               ibegin=cells(8,icell)
               isize=cells(9,icell)
               call L2P(wavek,scale(level),centers(1,icell),
     1              Local(iaddr(icell)),
     1              nterms(level),nterms_eval(1,level),nbessel,
     1              Xj(1,ibegin),isize,
     1              pot(ibegin),fld(1,ibegin),
     1              Anm1,Anm2,Pmax)
            endif
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'L2P    =',toc-tic

c     ... step 6: P2P
c$    tic=omp_get_wtime()
c$omp parallel do default(shared)
c$omp$private(icell,list,nlist)
c$omp$private(jcell,ilist)
c$omp$schedule(dynamic)
      do icell=1,numCells
         if (cells(7,icell).eq.0) then
c     ... evaluate self interactions
            call P2P(cells(1,icell),pot,fld,
     1           cells(1,icell),Xj,qj,wavek)
c     ... evaluate interactions with the nearest neighbours
            call getList(1,icell,list,nlist)
c     ... for all pairs in list #1, evaluate the potentials and fields directly
            do ilist=1,nlist
               jcell=list(ilist)
c     ... prune all sourceless cells
               if (cells(9,jcell).eq.0) cycle
               call P2P(cells(1,icell),pot,fld,
     1              cells(1,jcell),Xj,qj,wavek)
            enddo
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'P2P    =',toc-tic
      return
      end
