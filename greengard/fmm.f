      subroutine fmm(iprec,wavek,numBodies,Xj,qj,pi,Fi)
      use arrays, only : levelOffset,boxes
      use omp_lib, only : omp_get_wtime
      implicit none
      integer iprec,ncrit,nboxes,nlev,i,nmax,ibox,numBodies
      integer level,sumTerms
      integer nterms(0:200)
      integer, allocatable :: iaddr(:)
      integer, allocatable :: permutation(:)
      real *8 epsfmm,size,boxsize,tic/0.0d0/,toc/0.0d0/
      real *8 Xj(3,numBodies)
      real *8 Xjd(3,numBodies)
      real *8 bsize(0:200)
      real *8 scale(0:200)
      real *8 center(3)
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
      if( iprec .eq. -2 ) epsfmm=.5d-0
      if( iprec .eq. -1 ) epsfmm=.5d-1
      if( iprec .eq. 0 ) epsfmm=.5d-2
      if( iprec .eq. 1 ) epsfmm=.5d-3
      if( iprec .eq. 2 ) epsfmm=.5d-6
      if( iprec .eq. 3 ) epsfmm=.5d-9
      if( iprec .eq. 4 ) epsfmm=.5d-12
      if( iprec .eq. 5 ) epsfmm=.5d-15
      if( iprec .eq. 6 ) epsfmm=0
c     set criterion for box subdivision (number of sources per box)
      if( iprec .eq. -2 ) ncrit=40
      if( iprec .eq. -1 ) ncrit=50
      if( iprec .eq. 0 ) ncrit=80
      if( iprec .eq. 1 ) ncrit=160
      if( iprec .eq. 2 ) ncrit=400
      if( iprec .eq. 3 ) ncrit=800
      if( iprec .eq. 4 ) ncrit=1200
      if( iprec .eq. 5 ) ncrit=1400
      if( iprec .eq. 6 ) ncrit=numBodies
c     create oct-tree data structure
      allocate (permutation(numBodies))
      allocate (levelOffset(200)) 
c$    tic=omp_get_wtime() 
      call buildTree(Xj,numBodies,ncrit,
     1     nboxes,permutation,nlev,center,size)
      allocate(iaddr(nboxes))
      do i = 0,nlev
         scale(i) = 1.0d0
         boxsize = abs((size/2.0**i)*wavek)
         if (boxsize .lt. 1) scale(i) = boxsize
      enddo
      nmax = 0
      do i = 0,nlev
         bsize(i)=size/2.0d0**i
         call getNumTerms(1,1.5d0,bsize(i),wavek,epsfmm,nterms(i))
         if (nterms(i).gt. nmax .and. i.ge. 2) nmax = nterms(i)
      enddo
      do i = 1,numBodies
         Xjd(1,i) = Xj(1,permutation(i))
         Xjd(2,i) = Xj(2,permutation(i))
         Xjd(3,i) = Xj(3,permutation(i))
         qjd(i) = qj(permutation(i))
      enddo
      sumTerms=1
      do ibox=1,nboxes
         level=boxes(1,ibox)
         iaddr(ibox)=sumTerms
         sumTerms=sumTerms+(nterms(level)+1)*(2*nterms(level)+1)*2
      enddo
      allocate(Multipole(sumTerms))
      allocate(Local(sumTerms))
c$    toc=omp_get_wtime()
      print*,'Tree   =',toc-tic
      call evaluate(wavek,numBodies,Xjd,
     1     qjd,pid,Fid,epsfmm,iaddr,Multipole,Local,
     1     nboxes,nlev,scale,bsize,nterms)
      do i=1,numBodies
         pi(permutation(i))=pid(i)
         Fi(1,permutation(i))=Fid(1,i)
         Fi(2,permutation(i))=Fid(2,i)
         Fi(3,permutation(i))=Fid(3,i)
      enddo
      return
      end

      subroutine evaluate(wavek,
     1     numBodies,sourcesort,chargesort,pot,fld,
     1     epsfmm,iaddr,Multipole,Local,
     1     nboxes,nlev,scale,bsize,nterms)
      use arrays, only : listOffset,lists,levelOffset,boxes
      use omp_lib, only : omp_get_wtime
      implicit none
      integer Pmax,i,numBodies,itype,nlev,ibox,ilev
      integer nquad,level,level0,level1,ilist,nbessel
      integer jbox,nlist,nterms_trunc,ii,jj,kk
      integer nboxes
      integer box(20),box1(20),iaddr(nboxes),nterms(0:200),list(10000)
      integer itable(-3:3,-3:3,-3:3)
      integer nterms_eval(4,0:200)
      real *8 epsfmm,radius,tic/0.0d0/,toc/0.0d0/
      real *8 center0(3),corners0(3,8)
      real *8 center1(3),corners1(3,8)
      real *8 sourcesort(3,1)
      real *8 Multipole(1),Local(1),xquad(10000),wquad(10000)
      real *8 scale(0:200),bsize(0:200)
      real *8 Anm1(0:200,0:200)
      real *8 Anm2(0:200,0:200)
      complex *16 wavek
      complex *16 pot(1)
      complex *16 fld(3,1)
      complex *16 chargesort(1)
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
      do ibox = 1,nboxes
         level=boxes(1,ibox)
         call initCoefs(Multipole(iaddr(ibox)),nterms(level))
         call initCoefs(Local(iaddr(ibox)),nterms(level))
      enddo

c     ... step 1: P2M
c$    tic=omp_get_wtime()
      do ilev=3,nlev+1
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getCenter(ibox,center0,corners0)
            level=box(1)
            nbessel = nterms(level)+1000
            if(box(9).eq.0) cycle
            if(box(7).eq.0) then
               call P2M(wavek,scale(level),
     1              sourcesort(1,box(8)),chargesort(box(8)),box(9),
     1              center0,nterms(level),nterms_eval(1,level),nbessel,
     1              Multipole(iaddr(ibox)),Anm1,Anm2,Pmax)
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
c$omp$private(ibox,box,center0,corners0,level0)
c$omp$private(level,radius)
c$omp$private(i,jbox,box1,center1,corners1,level1)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getCenter(ibox,center0,corners0)
            if(box(9).eq.0) cycle
            if(box(7).ne.0) then
               level0=box(1)
               if(level0.ge.2) then
                  radius = (corners0(1,1) - center0(1))**2
                  radius = radius + (corners0(2,1) - center0(2))**2
                  radius = radius + (corners0(3,1) - center0(3))**2
                  radius = sqrt(radius)
                  do i=1,box(7)
                     jbox=box(6)+i-1
                     call getCell(jbox,box1,nboxes,center1,corners1)
                     call getCenter(jbox,center1,corners1)
                     level1=box1(1)
                     call M2M(wavek,scale(level1),center1,
     1                    Multipole(iaddr(jbox)),nterms(level1),
     1                    scale(level0),center0,
     1                    Multipole(iaddr(ibox)),
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
c$omp$private(ibox,box,center0,corners0,level0,list,ilist,nlist)
c$omp$private(jbox,box1,center1,corners1,level1,radius)
c$omp$private(nterms_trunc,ii,jj,kk)
c$omp$schedule(dynamic)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getCenter(ibox,center0,corners0)
            level0=box(1)
            if (level0 .ge. 2) then
               call getList(2,ibox,list,nlist)
               do ilist=1,nlist
                  jbox=list(ilist)
                  call getCell(jbox,box1,nboxes,center1,corners1)
                  call getCenter(jbox,center1,corners1)
                  if (box1(9).eq.0) cycle
                  radius = (corners1(1,1) - center1(1))**2
                  radius = radius + (corners1(2,1) - center1(2))**2
                  radius = radius + (corners1(3,1) - center1(3))**2
                  radius = sqrt(radius)
                  level1=box1(1)
                  ii=box1(2)-box(2)
                  jj=box1(3)-box(3)
                  kk=box1(4)-box(4)
                  nterms_trunc=itable(ii,jj,kk)
                  nterms_trunc=min(nterms(level0),nterms_trunc)
                  nterms_trunc=min(nterms(level1),nterms_trunc)
                  nbessel = nterms_trunc+1000
                  call M2L(wavek,
     1                 scale(level1),
     1                 center1,Multipole(iaddr(jbox)),
     1                 nterms(level1),scale(level0),
     1                 center0,Local(iaddr(ibox)),
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
c$omp$private(ibox,box,center0,corners0,level0)
c$omp$private(level,radius)
c$omp$private(i,jbox,box1,center1,corners1,level1)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getCenter(ibox,center0,corners0)
            if (box(7).ne.0) then
               level0=box(1)
               if (level0 .ge. 2) then
                  do i=1,box(7)
                     jbox=box(6)+i-1
                     if (jbox.eq.0) cycle
                     call getCell(jbox,box1,nboxes,center1,corners1)
                     call getCenter(jbox,center1,corners1)
                     radius = (corners1(1,1) - center1(1))**2
                     radius = radius + (corners1(2,1) - center1(2))**2
                     radius = radius + (corners1(3,1) - center1(3))**2
                     radius = sqrt(radius)
                     level1=box1(1)
                     nbessel = nquad+1000
                     call L2L(wavek,scale(level0),
     1                    center0,
     1                    Local(iaddr(ibox)),nterms(level0),
     1                    scale(level1),center1,Local(iaddr(jbox)),
     1                    nterms(level1),
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
c$omp$private(ibox,box,center0,corners0,level)
      do ibox=1,nboxes
         call getCell(ibox,box,nboxes,center0,corners0)
         call getCenter(ibox,center0,corners0)
         if (box(7).eq.0) then
            level=box(1)
            nbessel = nterms(level)+1000
            if (level .ge. 2) then
               call L2P(wavek,scale(level),center0,
     1              Local(iaddr(ibox)),
     1              nterms(level),nterms_eval(1,level),nbessel,
     1              sourcesort(1,box(8)),box(9),
     1              pot(box(8)),fld(1,box(8)),
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
c$omp$private(ibox,box,center0,corners0,list,nlist)
c$omp$private(jbox,box1,center1,corners1,ilist)
c$omp$schedule(dynamic)
      do ibox=1,nboxes
         if (boxes(7,ibox).eq.0) then
c     ... evaluate self interactions
            call P2P(boxes(1,ibox),sourcesort,pot,fld,
     1           boxes(1,ibox),sourcesort,chargesort,wavek)
c     ... evaluate interactions with the nearest neighbours
            call getList(1,ibox,list,nlist)
c     ... for all pairs in list #1, evaluate the potentials and fields directly
            do ilist=1,nlist
               jbox=list(ilist)
c     ... prune all sourceless boxes
               if( boxes(9,jbox) .eq. 0 ) cycle
               call P2P(boxes(1,ibox),sourcesort,pot,fld,
     1              boxes(1,jbox),sourcesort,chargesort,wavek)
            enddo
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'P2P    =',toc-tic
      return
      end
