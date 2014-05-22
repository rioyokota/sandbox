      subroutine fmm(ier,iprec,wavek,nsource,source,
     $     ifcharge,charge,pi,Fi)
      implicit real *8 (a-h,o-z)
      dimension source(3,nsource)
      complex *16 charge(nsource)
      complex *16 ima
      complex *16 pi(nsource)
      complex *16 Fi(3,nsource)
      complex *16 pid(nsource)
      complex *16 Fid(3,nsource)
      dimension timeinfo(10)
      dimension laddr(2,200)
      dimension bsize(0:200)
      dimension nterms(0:200)
      integer box(20)
      integer box1(20)
      dimension scale(0:200)
      dimension center(3)
      dimension center0(3),corners0(3,8)
      dimension center1(3),corners1(3,8)
      integer, allocatable :: iaddr(:)
      real *8, allocatable :: w(:)
      real *8, allocatable :: wlists(:)
      real *8, allocatable :: wrmlexp(:)
      complex *16 ptemp,ftemp(3)
      data ima/(0.0d0,1.0d0)/
      ier=0
      lused7 = 0
      pi=4*atan(1.0d0)
      ifprint=0
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
      if( iprec .eq. -2 ) nbox=40
      if( iprec .eq. -1 ) nbox=50
      if( iprec .eq. 0 ) nbox=80
      if( iprec .eq. 1 ) nbox=160
      if( iprec .eq. 2 ) nbox=400
      if( iprec .eq. 3 ) nbox=800
      if( iprec .eq. 4 ) nbox=1200
      if( iprec .eq. 5 ) nbox=1400
      if( iprec .eq. 6 ) nbox=nsource
c     create oct-tree data structure
      ntot = 100*nsource+10000
      do ii = 1,10
         allocate (wlists(ntot))
         call hfmm3dparttree(ier,iprec,
     1        nsource,source,
     1        nbox,epsfmm,iisource,iwlists,lwlists,
     1        nboxes,laddr,nlev,center,size,
     1        wlists,ntot,lused7)
         if (ier.eq.0) exit
         deallocate(wlists)
         ntot = ntot*1.5
      enddo
      allocate(iaddr(2*nboxes))
      lused7=1
      do i = 0,nlev
         scale(i) = 1.0d0
         boxsize = abs((size/2.0**i)*wavek)
         if (boxsize .lt. 1) scale(i) = boxsize
      enddo
c     isourcesort is pointer for sorted source coordinates
c     ichargesort is pointer for sorted charge densities
      isourcesort = lused7 + 5
      lsourcesort = 3*nsource
      ichargesort = isourcesort+lsourcesort
      lchargesort = 2*nsource
      lused7 = ichargesort+lchargesort
c     ... allocate the potential and field arrays
      ipot = lused7
      lpot = 2*nsource
      lused7=lused7+lpot
      ifld = lused7
      lfld = 2*(3*nsource)
      lused7=lused7+lfld
      ifldtarg = lused7
c     based on FMM tolerance, compute expansion lengths nterms(i)
      nmax = 0
      do i = 0,nlev
         bsize(i)=size/2.0d0**i
         call h3dterms(bsize(i),wavek,epsfmm, nterms(i), ier)
         if (nterms(i).gt. nmax .and. i.ge. 2) nmax = nterms(i)
      enddo
c     Multipole and local expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).

c     iiaddr is pointer to iaddr array, itself contained in workspace.
c     imptemp is pointer for single expansion (dimensioned by nmax)
      iiaddr = lused7
      imptemp = iiaddr + 2*nboxes
      lmptemp = (nmax+1)*(2*nmax+1)*2
      lused7 = imptemp + lmptemp
      allocate(w(lused7),stat=ier)
      call h3dreorder(nsource,source,ifcharge,charge,wlists(iisource),
     1     w(isourcesort),w(ichargesort))
      ifinit=1
      call h3dmpalloc(wlists(iwlists),iaddr,nboxes,lmptot,nterms)
      irmlexp = 1
      lused7 = irmlexp + lmptot
      allocate(wrmlexp(lused7),stat=ier)
      ifevalfar=1
      ifevalloc=1
      call evaluate(ier,iprec,wavek,
     1     ifevalfar,ifevalloc,
     1     nsource,w(isourcesort),wlists(iisource),
     1     ifcharge,w(ichargesort),pid,Fid,
     1     epsfmm,iaddr,wrmlexp(irmlexp),
     1     nboxes,laddr,nlev,scale,bsize,nterms,
     1     wlists(iwlists),lwlists)
      call h3dpsort(nsource,wlists(iisource),pid,pi)
      call h3dfsort(nsource,wlists(iisource),Fid,Fi)

      return
      end

      subroutine evaluate(ier,iprec,wavek,
     1     ifevalfar,ifevalloc,
     1     nsource,sourcesort,isource,
     1     ifcharge,chargesort,pot,fld,
     1     epsfmm,iaddr,rmlexp,
     1     nboxes,laddr,nlev,scale,bsize,nterms,
     1     wlists,lwlists)
      implicit real *8 (a-h,o-z)
      dimension sourcesort(3,1),isource(1)
      complex *16 chargesort(1),wavek
      complex *16 ima
      complex *16 pot(1)
      complex *16 fld(3,1)
      dimension wlists(1)
      dimension iaddr(2,nboxes)
      real *8 rmlexp(1),xnodes(10000),wts(10000)
      dimension center(3)
      dimension laddr(2,200)
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
      dimension Anm(100 000)
      dimension nterms_eval(4,0:200)
      complex *16 pottarg(1),fldtarg(3,1)
      real *8, allocatable :: rotmatf(:,:,:,:)
      real *8, allocatable :: rotmatb(:,:,:,:)
      real *8, allocatable :: thetas(:,:,:)
      real *8 rvec(3)
      data ima/(0.0d0,1.0d0)/
c     ... set the potential and field to zero
      do i=1,nsource
         pot(i)=0
         fld(1,i)=0
         fld(2,i)=0
         fld(3,i)=0
      enddo
c     ... initialize Legendre function evaluation routines
      Pmax=200
      lw7=100 000
      call ylgndrfwini(Pmax,Anm,lw7,lused7)
      do i=0,nlev
         do itype=1,4
            call h3dterms_eval(itype,bsize(i),wavek,epsfmm,
     1           nterms_eval(itype,i),ier)
         enddo
      enddo
c     ... set all multipole and local expansions to zero
      do ibox = 1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         level=box(1)
         call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level))
         call h3dzero(rmlexp(iaddr(2,ibox)),nterms(level))
      enddo

c     ... step 1: P2M
c$    tic=omp_get_wtime()
      do ilev=3,nlev+1
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level,npts,nkids,radius)
c$omp$private(lused,ier,i,j,ptemp,ftemp,cd)
         do ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
            call d3tgetb(ier,ibox,box,center0,corners0,wlists)
            call d3tnkids(box,nkids)
            level=box(1)
            nbessel = nterms(level)+1000
            if( box(15) .eq. 0 ) cycle
            if (nkids .eq. 0 ) then
               radius = (corners0(1,1) - center0(1))**2
               radius = radius + (corners0(2,1) - center0(2))**2
               radius = radius + (corners0(3,1) - center0(3))**2
               radius = sqrt(radius)
               call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level))
               call P2M(ier,wavek,scale(level),
     1              sourcesort(1,box(14)),chargesort(box(14)),box(15),
     1              center0,nterms(level),nterms_eval(1,level),nbessel,
     1              rmlexp(iaddr(1,ibox)),Anm,Pmax)
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
         call legewhts(nquad,xnodes,wts,1)
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level0,level,npts,nkids,radius)
c$omp$private(jbox,box1,center1,corners1,level1)
c$omp$private(lused,ier,i,j,ptemp,ftemp,cd)
         do ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
            call d3tgetb(ier,ibox,box,center0,corners0,wlists)
            call d3tnkids(box,nkids)
            if( box(15) .eq. 0 ) cycle
            if (nkids .ne. 0) then
               level0=box(1)
               if( level0 .ge. 2 ) then
                  radius = (corners0(1,1) - center0(1))**2
                  radius = radius + (corners0(2,1) - center0(2))**2
                  radius = radius + (corners0(3,1) - center0(3))**2
                  radius = sqrt(radius)
                  call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level0))
                  do i = 1,8
                     jbox = box(5+i)
                     if (jbox.eq.0) cycle
                     call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
                     level1=box1(1)
                     call M2M(wavek,scale(level1),center1,
     1                    rmlexp(iaddr(1,jbox)),nterms(level1),
     1                    scale(level0),center0,rmlexp(iaddr(1,ibox)),
     1                    nterms(level0),
     1                    radius,xnodes,wts,nquad,ier)
                  enddo
               endif
            endif
         enddo
c$omp end parallel do
      enddo
c$    toc=omp_get_wtime()
      print*,'M2M    =',toc-tic

c     ... step 3, precompute rotation matrices
c     (approximately 30kB of storage for ldm=10)
c     (approximately 40MB of storage for ldm=30)
c$    tic=omp_get_wtime()
      ldm = 1
      do i=2,nlev
         if( nterms(i) .gt. ldm) ldm = nterms(i)
      enddo
      if( ldm .gt. 10 ) ldm = 10
      allocate(rotmatf((ldm+1)*(ldm+1)*(2*ldm+1),-3:3,-3:3,-3:3))
      allocate(rotmatb((ldm+1)*(ldm+1)*(2*ldm+1),-3:3,-3:3,-3:3))
      allocate(thetas(-3:3,-3:3,-3:3))
      nstor = (ldm+1)*(ldm+1)*(2*ldm+1)*7*7*7  * 2
      thetas(0,0,0)=0
      do i=-3,3
         do j=-3,3
            do k=-3,3
               if( abs(i).gt.0 .or. abs(j).gt.0 .or. abs(k).gt.0 ) then
                  rvec(1) = i
                  rvec(2) = j
                  rvec(3) = k
                  call cart2sph(rvec,d,theta,phi)
                  thetas(i,j,k)=theta
                  call rotviarecur3p_init(ier,rotmatf(1,i,j,k),
     1                 ldm,+theta)
                  call rotviarecur3p_init(ier,rotmatb(1,i,j,k),
     1                 ldm,-theta)
               endif
            enddo
         enddo
      enddo
c     ... step 4, M2L
c$    tic=omp_get_wtime()
      do 4300 ilev=3,nlev+1
         call h3dterms_list2(bsize(ilev-1),wavek,epsfmm, itable, ier)
         nquad=nterms(ilev-1)*1.2
         nquad=max(6,nquad)
         call legewhts(nquad,xnodes,wts,1)
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,list,nlist)
c$omp$private(jbox,box1,center1,corners1,level1,ifdirect2,radius)
c$omp$private(lused,ier,i,j,ptemp,ftemp,cd,ilist,itype)
c$omp$private(nterms_trunc,ii,jj,kk)
c$omp$schedule(dynamic)
         do 4200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
            call d3tgetb(ier,ibox,box,center0,corners0,wlists)
            level0=box(1)
            if (level0 .ge. 2) then
c     ... retrieve list #2
               itype=2
               call d3tgetl(ier,ibox,itype,list,nlist,wlists)
c     ... for all pairs in list #2, apply the translation operator
               do 4150 ilist=1,nlist
                  jbox=list(ilist)
                  call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
                  if (box1(15).eq.0) goto 4150
                  if ((box(17).eq.0).and.(ifprune_list2.eq.1))
     $                 goto 4150
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
     1                 center1,rmlexp(iaddr(1,jbox)),
     1                 nterms(level1),scale(level0),
     1                 center0,rmlexp(iaddr(2,ibox)),
     1                 nterms(level0),nterms_trunc,
     1                 radius,xnodes,wts,nquad,nbessel,ier)
 4150          continue
            endif
 4200    continue
 4300 continue
c$    toc=omp_get_wtime()
      print*,'M2L    =',toc-tic

c     ... step 5, L2L
c$    tic=omp_get_wtime()
      do 5300 ilev=3,nlev
         nquad=nterms(ilev-1)*2
         nquad=max(6,nquad)
         call legewhts(nquad,xnodes,wts,1)
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level0,level,npts,nkids,radius)
c$omp$private(jbox,box1,center1,corners1,level1)
c$omp$private(lused,ier,i,j,ptemp,ftemp,cd)
         do 5200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
            call d3tgetb(ier,ibox,box,center0,corners0,wlists)
            call d3tnkids(box,nkids)
            if (nkids .ne. 0) then
               level0=box(1)
               if (level0 .ge. 2) then
c     ... split local expansion of the parent box
                  do 5100 i = 1,8
                     jbox = box(5+i)
                     if (jbox.eq.0) goto 5100
                     call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
                     radius = (corners1(1,1) - center1(1))**2
                     radius = radius + (corners1(2,1) - center1(2))**2
                     radius = radius + (corners1(3,1) - center1(3))**2
                     radius = sqrt(radius)
                     level1=box1(1)
                     nbessel = nquad+1000
                     call L2L(wavek,scale(level0),
     1                    center0,
     1                    rmlexp(iaddr(2,ibox)),nterms(level0),
     1                    scale(level1),center1,rmlexp(iaddr(2,jbox)),
     1                    nterms(level1),
     1                    radius,xnodes,wts,nquad,nbessel,ier)
 5100             continue
               endif
            endif
 5200    continue
c$omp end parallel do
 5300 continue
c$    toc=omp_get_wtime()
      print*,'L2L    =',toc-tic

c     ... step 6: L2P
c$    tic=omp_get_wtime()
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,level,npts,nkids,ier)
      do ibox=1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         call d3tnkids(box,nkids)
         if (nkids .eq. 0) then
            level=box(1)
            npts=box(15)
            nbessel = nterms(level)+1000
            if (level .ge. 2) then
               call L2P(wavek,scale(level),center0,
     1              rmlexp(iaddr(2,ibox)),
     1              nterms(level),nterms_eval(1,level),nbessel,
     1              sourcesort(1,box(14)),box(15),
     1              pot(box(14)),fld(1,box(14)),
     1              Anm,Pmax,ier)
            endif
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'L2P    =',toc-tic

c     ... step 8: P2P
c$    tic=omp_get_wtime()
c$omp parallel do default(shared)
c$omp$private(ibox,box,center0,corners0,nkids,list,nlist,npts)
c$omp$private(jbox,box1,center1,corners1,ier,ilist,itype)
c$omp$schedule(dynamic)
      do ibox=1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         call d3tnkids(box,nkids)
         if (nkids .eq. 0) then
c     ... evaluate self interactions
            call P2P(box,sourcesort,pot,fld,
     1           box,sourcesort,chargesort,wavek)
c     ... evaluate interactions with the nearest neighbours
            itype=1
            call d3tgetl(ier,ibox,itype,list,nlist,wlists)
c     ... for all pairs in list #1, evaluate the potentials and fields directly
            do ilist=1,nlist
               jbox=list(ilist)
               call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
c     ... prune all sourceless boxes
               if( box1(15) .eq. 0 ) cycle
               call P2P(box,sourcesort,pot,fld,
     1              box1,sourcesort,chargesort,wavek)
            enddo
         endif
      enddo
c$omp end parallel do
c$    toc=omp_get_wtime()
      print*,'P2P    =',toc-tic
      return
      end
