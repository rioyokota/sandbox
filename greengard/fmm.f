      subroutine fmm(iprec,wavek,numBodies,Xj,
     $     qj,pi,Fi)
      use arrays, only : levelOffset
      implicit real *8 (a-h,o-z)
      integer box(20)
      dimension Xj(3,numBodies)
      dimension Xjd(3,numBodies)
      complex *16 qj(numBodies)
      complex *16 qjd(2*numBodies)
      complex *16 imag
      complex *16 pi(numBodies)
      complex *16 Fi(3,numBodies)
      complex *16 pid(numBodies)
      complex *16 Fid(3,numBodies)
      dimension bsize(0:200)
      dimension nterms(0:200)
      dimension scale(0:200)
      dimension center(3)
      dimension center0(3)
      dimension corners0(3,8)
      integer, allocatable :: iaddr(:)
      integer, allocatable :: isource(:)
      real *8, allocatable :: Multipole(:)
      real *8, allocatable :: Local(:)
      complex *16 ptemp,ftemp(3)
      data imag/(0.0d0,1.0d0)/, tic/0.0d0/, toc/0.0d0/
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
      allocate (isource(numBodies))
      allocate (levelOffset(200)) 
c$    tic=omp_get_wtime() 
      call buildTree(Xj,numBodies,ncrit,
     1     nboxes,isource,nlev,center,size)
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
         Xjd(1,i) = Xj(1,isource(i))
         Xjd(2,i) = Xj(2,isource(i))
         Xjd(3,i) = Xj(3,isource(i))
         qjd(i) = qj(isource(i))
      enddo
      ifinit=1
      iptr=1
      do ibox=1,nboxes
         call getCell(ibox,box,nboxes,center0,corners0)
         level=box(1)
         iaddr(ibox)=iptr
         iptr=iptr+(nterms(level)+1)*(2*nterms(level)+1)*2
      enddo
      lmptot = iptr
      allocate(Multipole(lmptot))
      allocate(Local(lmptot))
c$    toc=omp_get_wtime()
      print*,'Tree   =',toc-tic
      call evaluate(iprec,wavek,numBodies,Xjd,isource,
     1     1,qjd,pid,Fid,epsfmm,iaddr,Multipole,Local,
     1     nboxes,nlev,scale,bsize,nterms)
      do i=1,numBodies
         pi(isource(i))=pid(i)
         Fi(1,isource(i))=Fid(1,i)
         Fi(2,isource(i))=Fid(2,i)
         Fi(3,isource(i))=Fid(3,i)
      enddo
      return
      end

      subroutine evaluate(iprec,wavek,
     1     numBodies,sourcesort,isource,
     1     ifcharge,chargesort,pot,fld,
     1     epsfmm,iaddr,Multipole,Local,
     1     nboxes,nlev,scale,bsize,nterms)
      use arrays, only : listOffset,lists,levelOffset
      implicit real *8 (a-h,o-z)
      dimension sourcesort(3,1),isource(1)
      complex *16 chargesort(1),wavek
      complex *16 imag
      complex *16 pot(1)
      complex *16 fld(3,1)
      dimension iaddr(nboxes)
      real *8 Multipole(1),Local(1),xquad(10000),wquad(10000)
      dimension center(3)
      dimension scale(0:200)
      dimension bsize(0:200)
      dimension nterms(0:200)
      dimension list(10 000)
      complex *16 ptemp,ftemp(3)
      integer Pmax,box(20)
      dimension center0(3),corners0(3,8)
      integer box1(20)
      dimension center1(3),corners1(3,8)
      dimension itable(-3:3,-3:3,-3:3)
      dimension Anm1(0:200,0:200)
      dimension Anm2(0:200,0:200)
      dimension nterms_eval(4,0:200)
      complex *16 pottarg(1),fldtarg(3,1)
      real *8, allocatable :: rotmatf(:,:,:,:)
      real *8, allocatable :: rotmatb(:,:,:,:)
      real *8, allocatable :: thetas(:,:,:)
      real *8 rvec(3)
      data imag/(0.0d0,1.0d0)/, tic/0.0d0/, toc/0.0d0/
c     ... set the potential and field to zero
      do i=1,numBodies
         pot(i)=0
         fld(1,i)=0
         fld(2,i)=0
         fld(3,i)=0
      enddo
c     ... initialize Legendre function evaluation routines
      Pmax=200
      lw7=100 000
      call getAnm(Pmax,Anm1,Anm2)
      do i=0,nlev
         do itype=1,4
            call getNumTerms(itype,1.5d0,bsize(i),wavek,epsfmm,
     1           nterms_eval(itype,i))
         enddo
      enddo
c     ... set all multipole and local expansions to zero
      do ibox = 1,nboxes
         call getCell(ibox,box,nboxes,center0,corners0)
         level=box(1)
         call initCoefs(Multipole(iaddr(ibox)),nterms(level))
         call initCoefs(Local(iaddr(ibox)),nterms(level))
      enddo

c     ... step 1: P2M
c$    tic=omp_get_wtime()
      do ilev=3,nlev+1
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level,npts,numChild,radius)
c$omp$private(i,j,ptemp,ftemp,cd)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getNumChild(box,numChild)
            level=box(1)
            nbessel = nterms(level)+1000
            if( box(15) .eq. 0 ) cycle
            if (numChild .eq. 0 ) then
               radius = (corners0(1,1) - center0(1))**2
               radius = radius + (corners0(2,1) - center0(2))**2
               radius = radius + (corners0(3,1) - center0(3))**2
               radius = sqrt(radius)
               call P2M(wavek,scale(level),
     1              sourcesort(1,box(14)),chargesort(box(14)),box(15),
     1              center0,nterms(level),nterms_eval(1,level),nbessel,
     1              Multipole(iaddr(ibox)),Anm1,Anm2,Pmax)
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'P2M    =',toc-tic

      ifprune_list2 = 0
c     ... step 2, M2M
c$    tic=omp_get_wtime()
      do ilev=nlev,3,-1
         nquad=nterms(ilev-1)*2.5
         nquad=max(6,nquad)
         call legendre(nquad,xquad,wquad)
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level0)
c$omp$private(level,npts,numChild,radius)
c$omp$private(jbox,box1,center1,corners1,level1)
c$omp$private(i,j,ptemp,ftemp,cd)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getNumChild(box,numChild)
            if( box(15) .eq. 0 ) cycle
            if (numChild .ne. 0) then
               level0=box(1)
               if( level0 .ge. 2 ) then
                  radius = (corners0(1,1) - center0(1))**2
                  radius = radius + (corners0(2,1) - center0(2))**2
                  radius = radius + (corners0(3,1) - center0(3))**2
                  radius = sqrt(radius)
                  do i = 1,8
                     jbox = box(5+i)
                     if (jbox.eq.0) cycle
                     call getCell(jbox,box1,nboxes,center1,corners1)
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
c$omp$private(ibox,box,center0,corners0,list,nlist)
c$omp$private(jbox,box1,center1,corners1,level1,ifdirect2,radius)
c$omp$private(i,j,ptemp,ftemp,cd,ilist,itype)
c$omp$private(nterms_trunc,ii,jj,kk)
c$omp$schedule(dynamic)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            level0=box(1)
            if (level0 .ge. 2) then
c     ... retrieve list #2
               itype=2
               call getList(itype,ibox,nboxes,list,nlist)
c     ... for all pairs in list #2, apply the translation operator
               do ilist=1,nlist
                  jbox=list(ilist)
                  call getCell(jbox,box1,nboxes,center1,corners1)
                  if (box1(15).eq.0) cycle
                  if ((box(17).eq.0).and.(ifprune_list2.eq.1)) cycle
                  radius = (corners1(1,1) - center1(1))**2
                  radius = radius + (corners1(2,1) - center1(2))**2
                  radius = radius + (corners1(3,1) - center1(3))**2
                  radius = sqrt(radius)
c     ... convert multipole expansions for all boxes in list 2 in local exp
c     ... if source is childless, evaluate directly (if cheaper)
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
c$omp$private(level,npts,numChild,radius)
c$omp$private(jbox,box1,center1,corners1,level1)
c$omp$private(i,j,ptemp,ftemp,cd)
         do ibox=levelOffset(ilev),levelOffset(ilev+1)-1
            call getCell(ibox,box,nboxes,center0,corners0)
            call getNumChild(box,numChild)
            if (numChild .ne. 0) then
               level0=box(1)
               if (level0 .ge. 2) then
c     ... split local expansion of the parent box
                  do i = 1,8
                     jbox = box(5+i)
                     if (jbox.eq.0) cycle
                     call getCell(jbox,box1,nboxes,center1,corners1)
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
c$omp$private(ibox,box,center0,corners0,level,npts,numChild)
      do ibox=1,nboxes
         call getCell(ibox,box,nboxes,center0,corners0)
         call getNumChild(box,numChild)
         if (numChild .eq. 0) then
            level=box(1)
            npts=box(15)
            nbessel = nterms(level)+1000
            if (level .ge. 2) then
               call L2P(wavek,scale(level),center0,
     1              Local(iaddr(ibox)),
     1              nterms(level),nterms_eval(1,level),nbessel,
     1              sourcesort(1,box(14)),box(15),
     1              pot(box(14)),fld(1,box(14)),
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
c$omp$private(ibox,box,center0,corners0,numChild,list,nlist,npts)
c$omp$private(jbox,box1,center1,corners1,ilist,itype)
c$omp$schedule(dynamic)
      do ibox=1,nboxes
         call getCell(ibox,box,nboxes,center0,corners0)
         call getNumChild(box,numChild)
         if (numChild .eq. 0) then
c     ... evaluate self interactions
            call P2P(box,sourcesort,pot,fld,
     $           box,sourcesort,chargesort,wavek)
c     ... evaluate interactions with the nearest neighbours
            itype=1
            call getList(itype,ibox,nboxes,list,nlist)
c     ... for all pairs in list #1, evaluate the potentials and fields directly
            do ilist=1,nlist
               jbox=list(ilist)
               call getCell(jbox,box1,nboxes,center1,corners1)
c     ... prune all sourceless boxes
               if( box1(15) .eq. 0 ) cycle
               call P2P(box,sourcesort,pot,fld,
     $              box1,sourcesort,chargesort,wavek)
            enddo
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'P2P    =',toc-tic
      return
      end
