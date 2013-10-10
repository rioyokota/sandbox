      subroutine fmm(ier,iprec,zk,nsource,source,
     $     ifcharge,charge,pot,fld)
      implicit real *8 (a-h,o-z)
      dimension source(3,nsource)
      complex *16 charge(nsource)
      complex *16 ima
      complex *16 pot(nsource)
      complex *16 fld(3,nsource)
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
         call hfmm3dparttree(ier,iprec,zk,
     1        nsource,source,
     1        nbox,epsfmm,iisource,iwlists,lwlists,
     1        nboxes,laddr,nlev,center,size,
     1        wlists,ntot,lused7)
         if (ier.eq.0) exit
         deallocate(wlists)
         ntot = ntot*1.5
      enddo
      lused7=1
      do i = 0,nlev
         scale(i) = 1.0d0
         boxsize = abs((size/2.0**i)*zk)
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
         call h3dterms(bsize(i),zk,epsfmm, nterms(i), ier)
         if (nterms(i).gt. nmax .and. i.ge. 2) nmax = nterms(i)
      enddo
      nquad=2*nmax               
c     ixnodes is pointer for quadrature nodes
c     iwhts is pointer for quadrature weights
      ixnodes = lused7 
      iwts = ixnodes + nquad
      lused7 = iwts + nquad
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
      call legewhts(nquad,w(ixnodes),w(iwts),ifinit)
      call h3dmpalloc(wlists(iwlists),w(iiaddr),nboxes,lmptot,nterms)
      irmlexp = 1
      lused7 = irmlexp + lmptot 
      allocate(wrmlexp(lused7),stat=ier)
      ifevalfar=1
      ifevalloc=1
      call hfmm3dparttargmain(ier,iprec,zk,
     1     ifevalfar,ifevalloc,
     1     nsource,w(isourcesort),wlists(iisource),
     1     ifcharge,w(ichargesort),w(ipot),w(ifld),
     1     epsfmm,w(iiaddr),wrmlexp(irmlexp),w(imptemp),lmptemp,
     1     w(ixnodes),w(iwts),nquad,
     1     nboxes,laddr,nlev,scale,bsize,nterms,
     1     wlists(iwlists),lwlists)
      call h3dpsort(nsource,wlists(iisource),w(ipot),pot)
      call h3dfsort(nsource,wlists(iisource),w(ifld),fld)

      return
      end

      subroutine hfmm3dparttargmain(ier,iprec,zk,
     1     ifevalfar,ifevalloc,
     1     nsource,sourcesort,isource,
     1     ifcharge,chargesort,pot,fld,
     1     epsfmm,iaddr,rmlexp,mptemp,lmptemp,xnodes,wts,nquad,
     1     nboxes,laddr,nlev,scale,bsize,nterms,
     1     wlists,lwlists)
      implicit real *8 (a-h,o-z)
      dimension sourcesort(3,1), isource(1)
      complex *16 chargesort(1),zk
      complex *16 ima
      complex *16 pot(1)
      complex *16 fld(3,1)
      dimension wlists(1)
      dimension iaddr(2,nboxes)
      real *8 rmlexp(1)
      complex *16 mptemp(lmptemp)
      dimension xnodes(nquad),wts(nquad)
      dimension center(3)
      dimension laddr(2,200)
      dimension scale(0:200)
      dimension bsize(0:200)
      dimension nterms(0:200)
      dimension list(10 000)
      complex *16 ptemp,ftemp(3)
      integer box(20)
      dimension center0(3),corners0(3,8)
      integer box1(20)
      dimension center1(3),corners1(3,8)
      dimension itable(-3:3,-3:3,-3:3)
      dimension wlege(100 000)
      dimension nterms_eval(4,0:200)
      data ima/(0.0d0,1.0d0)/
c     ... set the potential and field to zero
      do i=1,nsource
         pot(i)=0
         fld(1,i)=0
         fld(2,i)=0
         fld(3,i)=0
      enddo
c     ... initialize Legendre function evaluation routines
      nlege=200
      lw7=100 000
      call ylgndrfwini(nlege,wlege,lw7,lused7)
      do i=0,nlev
         do itype=1,4
            call h3dterms_eval(itype,bsize(i),zk,epsfmm,
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

      t1=omp_get_wtime()
c     ... step 1, locate all charges, assign them to boxes, and
c     form multipole expansions
      do ilev=3,nlev+1
c$OMP PARALLEL DO DEFAULT(SHARED)
c$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,radius)
c$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd) 
         do ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
            call d3tgetb(ier,ibox,box,center0,corners0,wlists)
            call d3tnkids(box,nkids)
            level=box(1)
            if (nkids .eq. 0) then
               npts=box(15)
            endif
c     ... prune all sourceless boxes
            if( box(15) .eq. 0 ) cycle
            if (nkids .eq. 0 ) then
c     ... form multipole expansions
               radius = (corners0(1,1) - center0(1))**2
               radius = radius + (corners0(2,1) - center0(2))**2
               radius = radius + (corners0(3,1) - center0(3))**2
               radius = sqrt(radius)
               call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level))
               if_use_trunc = 1
               call h3dformmp_add_trunc(ier,zk,scale(level),
     1              sourcesort(1,box(14)),chargesort(box(14)),npts,
     1              center0,nterms(level),nterms_eval(1,level),
     1              rmlexp(iaddr(1,ibox)),wlege,nlege)
            endif
         enddo
c$OMP END PARALLEL DO
      enddo
      t2=omp_get_wtime()

      ifprune_list2 = 0
      call hfmm3d_list2
     1     (zk,bsize,nlev,laddr,scale,nterms,rmlexp,iaddr,epsfmm,
     1     wlists,mptemp,lmptemp,xnodes,wts,nquad,
     1     ifprune_list2)

      t1=omp_get_wtime()
c     ... step 7, evaluate local expansions and all fields directly
c$OMP PARALLEL DO DEFAULT(SHARED)
c$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,ier)
      do ibox=1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         call d3tnkids(box,nkids)
         if (nkids .eq. 0) then
            level=box(1)
            npts=box(15)
            if (level .ge. 2) then
               call h3dtaevalall_trunc(zk,scale(level),center0,
     1              rmlexp(iaddr(2,ibox)),
     1              nterms(level),nterms_eval(1,level),
     1              sourcesort(1,box(14)),box(15),
     1              pot(box(14)),
     1              fld(1,box(14)),
     1              wlege,nlege,ier)
            endif
         endif
      enddo
c$OMP END PARALLEL DO
      t2=omp_get_wtime()

      t1=omp_get_wtime()
c     ... step 8, evaluate direct interactions 
c$OMP PARALLEL DO DEFAULT(SHARED)
c$OMP$PRIVATE(ibox,box,center0,corners0,nkids,list,nlist,npts)
c$OMP$PRIVATE(jbox,box1,center1,corners1)
c$OMP$PRIVATE(ier,ilist,itype) 
c$OMP$SCHEDULE(DYNAMIC)
      do ibox=1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         call d3tnkids(box,nkids)
         if (nkids .eq. 0) then
c     ... evaluate self interactions
            call hfmm3dpart_direct_self(zk,box,sourcesort,
     1           chargesort,pot,fld)
c     ... evaluate interactions with the nearest neighbours
            itype=1
            call d3tgetl(ier,ibox,itype,list,nlist,wlists)
c     ... for all pairs in list #1, evaluate the potentials and fields directly
            do ilist=1,nlist
               jbox=list(ilist)
               call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
c     ... prune all sourceless boxes
               if( box1(15) .eq. 0 ) cycle
               call hfmm3dpart_direct_targ(zk,box1,box,sourcesort,
     1              chargesort,pot,fld)
            enddo
         endif
      enddo
c$OMP END PARALLEL DO
      t2=omp_get_wtime()
      return
      end

      subroutine hfmm3dpart_direct_self(zk,box,
     1     source,charge,pot,fld)
      implicit real *8 (a-h,o-z)
      integer box(20)
      dimension source(3,1)
      complex *16 charge(1),zk
      complex *16 pot(1),fld(3,1)
      complex *16 ptemp,ftemp(3)
      do j=box(14),box(14)+box(15)-1
         do i=box(14),box(14)+box(15)-1
            if (i .eq. j) cycle
            call P2P(source(1,i),charge(i),
     1           source(1,j),zk,ptemp,ftemp)
            pot(j)=pot(j)+ptemp
            fld(1,j)=fld(1,j)+ftemp(1)
            fld(2,j)=fld(2,j)+ftemp(2)
            fld(3,j)=fld(3,j)+ftemp(3)
         enddo
      enddo
      return
      end

      subroutine hfmm3dpart_direct_targ(zk,box,box1,
     1     source,charge,pot,fld)
      implicit real *8 (a-h,o-z)
      integer box(20),box1(20)
      dimension source(3,1)
      complex *16 charge(1),zk
      complex *16 pot(1),fld(3,1)
      complex *16 ptemp,ftemp(3)
      do j=box1(14),box1(14)+box1(15)-1
         do i=box(14),box(14)+box(15)-1
            call P2P(source(1,i),charge(i),source(1,j),zk,
     1           ptemp,ftemp)
            pot(j)=pot(j)+ptemp
            fld(1,j)=fld(1,j)+ftemp(1)
            fld(2,j)=fld(2,j)+ftemp(2)
            fld(3,j)=fld(3,j)+ftemp(3)
         enddo
      enddo
      return
      end
