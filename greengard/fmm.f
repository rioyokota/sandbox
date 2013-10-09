        subroutine hfmm3dparttarg(ier,iprec,zk,nsource,source,
     $     ifcharge,charge,ifdipole,dipstr,dipvec,
     $     ifpot,pot,iffld,fld,
     $     ntarget,target,ifpottarg,pottarg,iffldtarg,fldtarg)
        implicit real *8 (a-h,o-z)
        dimension source(3,nsource)
        complex *16 charge(nsource)
        complex *16 dipstr(nsource)
        dimension dipvec(3,nsource)
        complex *16 ima
        complex *16 pot(nsource)
        complex *16 fld(3,nsource)
        dimension target(3,ntarget)
        complex *16 pottarg(ntarget)
        complex *16 fldtarg(3,ntarget)
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
c       
        data ima/(0.0d0,1.0d0)/
c       
        ier=0
        lused7 = 0
c       
        done=1
        pi=4*atan(done)
c
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c       
        ifprint=0
c
c     set fmm tolerance based on iprec flag.
c
        if( iprec .eq. -2 ) epsfmm=.5d-0 
        if( iprec .eq. -1 ) epsfmm=.5d-1
        if( iprec .eq. 0 ) epsfmm=.5d-2
        if( iprec .eq. 1 ) epsfmm=.5d-3
        if( iprec .eq. 2 ) epsfmm=.5d-6
        if( iprec .eq. 3 ) epsfmm=.5d-9
        if( iprec .eq. 4 ) epsfmm=.5d-12
        if( iprec .eq. 5 ) epsfmm=.5d-15
        if( iprec .eq. 6 ) epsfmm=0
c       
        if (ifprint .ge. 1) call prin2('epsfmm=*',epsfmm,1)
c
c
c     set criterion for box subdivision (number of sources per box)
c
        if( iprec .eq. -2 ) nbox=40
        if( iprec .eq. -1 ) nbox=50
        if( iprec .eq. 0 ) nbox=80
        if( iprec .eq. 1 ) nbox=160
        if( iprec .eq. 2 ) nbox=400
        if( iprec .eq. 3 ) nbox=800
        if( iprec .eq. 4 ) nbox=1200
        if( iprec .eq. 5 ) nbox=1400
        if( iprec .eq. 6 ) nbox=nsource+ntarget
c
        if (ifprint .ge. 1) call prinf('nbox=*',nbox,1)
c
c
c     create oct-tree data structure
c
        ntot = 100*(nsource+ntarget)+10000
        do ii = 1,10
           allocate (wlists(ntot))
           call hfmm3dparttree(ier,iprec,zk,
     $        nsource,source,ntarget,target,
     $        nbox,epsfmm,iisource,iitarget,iwlists,lwlists,
     $        nboxes,laddr,nlev,center,size,
     $        wlists,ntot,lused7)
           if (ier.ne.0) then
              deallocate(wlists)
              ntot = ntot*1.5
              call prinf(' increasing allocation, ntot is *',ntot,1)
           else
             goto 1200
           endif
        enddo
1200    continue
        if (ier.ne.0) then
           call prinf(' exceeded max allocation, ntot is *',ntot,1)
           ier = 4
           return
        endif
c
c     lused7 is counter that steps through workspace,
c     keeping track of total memory used.
c
        lused7=1
c
c       ... prepare data structures 
c
        do i = 0,nlev
        scale(i) = 1.0d0
        boxsize = abs((size/2.0**i)*zk)
        if (boxsize .lt. 1) scale(i) = boxsize
        enddo
c       
        if (ifprint .ge. 1) call prin2('scale=*',scale,nlev+1)
c       
c
c       carve up workspace further
c
c     isourcesort is pointer for sorted source coordinates
c     itargetsort is pointer for sorted target locations
c     ichargesort is pointer for sorted charge densities
c     idipvecsort is pointer for sorted dipole orientation vectors
c     idipstrsort is pointer for sorted dipole densities
c
        isourcesort = lused7 + 5
        lsourcesort = 3*nsource
        itargetsort = isourcesort+lsourcesort
        ltargetsort = 3*ntarget
        ichargesort = itargetsort+ltargetsort
        lchargesort = 2*nsource
        idipvecsort = ichargesort+lchargesort
        if (ifdipole.eq.1) then
          ldipvec = 3*nsource
          ldipstr = 2*nsource
        else
          ldipvec = 3
          ldipstr = 2
        endif
        idipstrsort = idipvecsort + ldipvec
        lused7 = idipstrsort + ldipstr
c
c       ... allocate the potential and field arrays
c
        ipot = lused7
        lpot = 2*nsource
        lused7=lused7+lpot
c       
        ifld = lused7
        if( iffld .eq. 1) then
        lfld = 2*(3*nsource)
        else
        lfld=6
        endif
        lused7=lused7+lfld
c      
        ipottarg = lused7
        lpottarg = 2*ntarget
        lused7=lused7+lpottarg
c       
        ifldtarg = lused7
        if( iffldtarg .eq. 1) then
        lfldtarg = 2*(3*ntarget)
        else
        lfldtarg=6
        endif
        lused7=lused7+lfldtarg
c      
        if (ifprint .ge. 1) call prinf(' lused7 is *',lused7,1)
c
c       based on FMM tolerance, compute expansion lengths nterms(i)
c      
        nmax = 0
        do i = 0,nlev
           bsize(i)=size/2.0d0**i
           call h3dterms(bsize(i),zk,epsfmm, nterms(i), ier)
           if (nterms(i).gt. nmax .and. i.ge. 2) nmax = nterms(i)
        enddo
c
        if (ifprint.eq.1) 
     $     call prin2('in hfmm3dpart, bsize(0) zk/2 pi=*',
     $     abs(bsize(0)*zk)/2/pi,1)
c
        if (ifprint.eq.1) call prin2('zk=*',zk,2)
        if (ifprint.eq.1) call prin2('bsize=*',bsize,nlev+1)
c
        nquad=2*nmax        
c       
c     ixnodes is pointer for quadrature nodes
c     iwhts is pointer for quadrature weights
c
        ixnodes = lused7 
        iwts = ixnodes + nquad
        lused7 = iwts + nquad
c
        if (ifprint .ge. 1) call prinf('nterms=*',nterms,nlev+1)
        if (ifprint .ge. 1) call prinf('nmax=*',nmax,1)
c
c     Multipole and local expansions will be held in workspace
c     in locations pointed to by array iaddr(2,nboxes).
c
c     iiaddr is pointer to iaddr array, itself contained in workspace.
c     imptemp is pointer for single expansion (dimensioned by nmax)
c   
c       ... allocate iaddr and temporary arrays
c
        iiaddr = lused7 
        imptemp = iiaddr + 2*nboxes
        lmptemp = (nmax+1)*(2*nmax+1)*2
        lused7 = imptemp + lmptemp
        allocate(w(lused7),stat=ier)
        if (ier.ne.0) then
           call prinf(' cannot allocate bulk FMM workspace,
     1                  lused7 is *',lused7,1)
           ier = 8
           return
        endif
c
c     reorder sources, targets so that each box holds
c     contiguous list of source/target numbers.
c
        call h3dreorder(nsource,source,ifcharge,charge,wlists(iisource),
     $     ifdipole,dipstr,dipvec,
     1     w(isourcesort),w(ichargesort),w(idipvecsort),w(idipstrsort)) 
c       
        call h3dreordertarg(ntarget,target,wlists(iitarget),
     1       w(itargetsort))
c
        if (ifprint .ge. 1) call prinf('finished reordering=*',ier,1)
        if (ifprint .ge. 1) call prinf('ier=*',ier,1)
        if (ifprint .ge. 1) call prinf('nboxes=*',nboxes,1)
        if (ifprint .ge. 1) call prinf('nlev=*',nlev,1)
        if (ifprint .ge. 1) call prinf('nboxes=*',nboxes,1)
        if (ifprint .ge. 1) call prinf('lused7=*',lused7,1)
c
        ifinit=1
        call legewhts(nquad,w(ixnodes),w(iwts),ifinit)
c
ccc        call prin2('xnodes=*',xnodes,nquad)
ccc        call prin2('wts=*',wts,nquad)

c     allocate memory need by multipole, local expansions at all
c     levels
c     irmlexp is pointer for workspace need by various fmm routines,
c
        call h3dmpalloc(wlists(iwlists),w(iiaddr),nboxes,lmptot,nterms)
c
        if (ifprint .ge. 1) call prinf(' lmptot is *',lmptot,1)
c       
        irmlexp = 1
        lused7 = irmlexp + lmptot 
        if (ifprint .ge. 1) call prinf(' lused7 is *',lused7,1)
        allocate(wrmlexp(lused7),stat=ier)
        if (ier.ne.0) then
           call prinf(' cannot allocate mpole expansion workspace,
     1                  lused7 is *',lused7,1)
           ier = 16
           return
        endif
c
c       
ccc        do i=lused7+1,lused7+1+100
ccc        w(i)=777
ccc        enddo
c
c     Memory allocation is complete. 
c     Call main fmm routine. There are, unfortunately, a lot
c     of parameters here. ifevalfar and ifevalloc determine
c     whether far field and local fields (respectively) are to 
c     be evaluated. Setting both to 1 means that both will be
c     computed (which is the normal scenario).
c
        ifevalfar=1
        ifevalloc=1
c
        call hfmm3dparttargmain(ier,iprec,zk,
     $     ifevalfar,ifevalloc,
     $     nsource,w(isourcesort),wlists(iisource),
     $     ifcharge,w(ichargesort),
     $     ifdipole,w(idipstrsort),w(idipvecsort),
     $     ifpot,w(ipot),iffld,w(ifld),
     $     ntarget,w(itargetsort),wlists(iitarget),
     $     ifpottarg,w(ipottarg),iffldtarg,w(ifldtarg),
     $     epsfmm,w(iiaddr),wrmlexp(irmlexp),w(imptemp),lmptemp,
     $     w(ixnodes),w(iwts),nquad,
     $     nboxes,laddr,nlev,scale,bsize,nterms,
     $     wlists(iwlists),lwlists)
c
c       parameter ier from targmain routine is currently meaningless, reset to 0
        if( ier .ne. 0 ) ier = 0
c
        if (ifprint .ge. 1) call prinf('lwlists=*',lused,1)
        if (ifprint .ge. 1) call prinf('lused total =*',lused7,1)       
c
        if (ifprint .ge. 1) 
     $      call prin2('memory / point = *',(lused7)/dble(nsource),1)
c       
ccc        call prin2('after w=*', w(1+lused7-100), 2*100)
c
        if(ifpot .eq. 1) 
     $     call h3dpsort(nsource,wlists(iisource),w(ipot),pot)
        if(iffld .eq. 1) 
     $     call h3dfsort(nsource,wlists(iisource),w(ifld),fld)
c
        if(ifpottarg .eq. 1 )
     $     call h3dpsort(ntarget,wlists(iitarget),w(ipottarg),pottarg)
        if(iffldtarg .eq. 1) 
     $     call h3dfsort(ntarget,wlists(iitarget),w(ifldtarg),fldtarg)
c       
        return
        end
c
c
c
c
c
        subroutine hfmm3dparttargmain(ier,iprec,zk,
     $     ifevalfar,ifevalloc,
     $     nsource,sourcesort,isource,
     $     ifcharge,chargesort,
     $     ifdipole,dipstrsort,dipvecsort,
     $     ifpot,pot,iffld,fld,ntarget,
     $     targetsort,itarget,ifpottarg,pottarg,iffldtarg,fldtarg,
     $     epsfmm,iaddr,rmlexp,mptemp,lmptemp,xnodes,wts,nquad,
     $     nboxes,laddr,nlev,scale,bsize,nterms,
     $     wlists,lwlists)
        implicit real *8 (a-h,o-z)
        dimension sourcesort(3,1), isource(1)
        complex *16 chargesort(1),zk
        complex *16 dipstrsort(1)
        dimension dipvecsort(3,1)
        complex *16 ima
        complex *16 pot(1)
        complex *16 fld(3,1)
        dimension targetsort(3,1), itarget(1)
        complex *16 pottarg(1)
        complex *16 fldtarg(3,1)
        dimension wlists(1)
        dimension iaddr(2,nboxes)
        real *8 rmlexp(1)
        complex *16 mptemp(lmptemp)
        dimension xnodes(nquad),wts(nquad)
        dimension timeinfo(10)
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
c
        data ima/(0.0d0,1.0d0)/

c
c
c     ifprint is an internal information printing flag. 
c     Suppressed if ifprint=0.
c     Prints timing breakdown and other things if ifprint=1.
c     Prints timing breakdown, list information, and other things if ifprint=2.
c       
        ifprint=0
c
c
c       ... set the potential and field to zero
c
        do i=1,nsource
        if( ifpot .eq. 1) pot(i)=0
        if( iffld .eq. 1) then
           fld(1,i)=0
           fld(2,i)=0
           fld(3,i)=0
        endif
        enddo
c       
        do i=1,ntarget
        if( ifpottarg .eq. 1) pottarg(i)=0
        if( iffldtarg .eq. 1) then
           fldtarg(1,i)=0
           fldtarg(2,i)=0
           fldtarg(3,i)=0
        endif
        enddo
c
        do i=1,10
        timeinfo(i)=0
        enddo
c       ... initialize Legendre function evaluation routines
c
        nlege=200
        lw7=100 000
        call ylgndrfwini(nlege,wlege,lw7,lused7)
ccc        write(*,*)' lused7 from  ylgndrfwini is',lused7
c
        do i=0,nlev
        do itype=1,4
        call h3dterms_eval(itype,bsize(i),zk,epsfmm,
     1       nterms_eval(itype,i),ier)
        enddo
        enddo
c
        if (ifprint .ge. 2) 
     $     call prinf('nterms_eval=*',nterms_eval,4*(nlev+1))
c
c       ... set all multipole and local expansions to zero
c
        do ibox = 1,nboxes
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        level=box(1)
        call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level))
        call h3dzero(rmlexp(iaddr(2,ibox)),nterms(level))
        enddo
c
c
        if(ifprint .ge. 1) 
     $     call prinf('=== STEP 1 (form mp) ====*',i,0)
        t1=omp_get_wtime()
c
c       ... step 1, locate all charges, assign them to boxes, and
c       form multipole expansions
c
ccc        do 1200 ibox=1,nboxes
        do 1300 ilev=3,nlev+1
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,radius)
C$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(1) 
        do 1200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
c
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        call d3tnkids(box,nkids)
c
        level=box(1)
c
c
        if (ifprint .ge. 2) then
           call prinf('ibox=*',ibox,1)
           call prinf('box=*',box,20)
           call prinf('nkids=*',nkids,1)
        endif
c
        if (nkids .eq. 0) then
c        ipts=box(14)
c        npts=box(15)
c        call prinf('ipts=*',ipts,1)
c        call prinf('npts=*',npts,1)
        npts=box(15)
        if (ifprint .ge. 2) then
           call prinf('npts=*',npts,1)
           call prinf('isource=*',isource(box(14)),box(15))
        endif
        endif
c
c       ... prune all sourceless boxes
c
        if( box(15) .eq. 0 ) goto 1200
c
        if (nkids .eq. 0) then
c
c       ... form multipole expansions
c
	    radius = (corners0(1,1) - center0(1))**2
	    radius = radius + (corners0(2,1) - center0(2))**2
	    radius = radius + (corners0(3,1) - center0(3))**2
	    radius = sqrt(radius)
c
            call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level))
            if_use_trunc = 1

            if( ifcharge .eq. 1 ) then
            call h3dformmp_add_trunc(ier,zk,scale(level),
     1         sourcesort(1,box(14)),chargesort(box(14)),npts,center0,
     $         nterms(level),nterms_eval(1,level),
     2         rmlexp(iaddr(1,ibox)),wlege,nlege)
            endif
c 
            if (ifdipole .eq. 1 ) then
            call h3dformmp_dp_add_trunc(ier,zk,scale(level),
     $         sourcesort(1,box(14)),
     1         dipstrsort(box(14)),dipvecsort(1,box(14)),
     $         npts,center0,nterms(level),nterms_eval(1,level),
     2         rmlexp(iaddr(1,ibox)),wlege,nlege)
            endif
         endif
c
 1200    continue
C$OMP END PARALLEL DO
 1300    continue
c
        t2=omp_get_wtime()
ccc        call prin2('time=*',t2-t1,1)
         timeinfo(1)=t2-t1
c       
         if(ifprint .ge. 1)
     $      call prinf('=== STEP 2 (form lo) ====*',i,0)
        t1=omp_get_wtime()
c
c       ... step 2, adaptive part, form local expansions, 
c           or evaluate the potentials and fields directly
c 
         do 3251 ibox=1,nboxes
c
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
c
         itype=3
         call d3tgetl(ier,ibox,itype,list,nlist,wlists)
         if (nlist .gt. 0) then 
            if (ifprint .ge. 2) then
               call prinf('ibox=*',ibox,1)
               call prinf('list3=*',list,nlist)
            endif
         endif
c
c       ... prune all sourceless boxes
c
         if( box(15) .eq. 0 ) nlist=0
c
c
c       ... note that lists 3 and 4 are dual
c
c       ... form local expansions for all boxes in list 3
c       ... if target is childless, evaluate directly (if cheaper)
c        
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(level,npts,nkids)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1,ifdirect3,radius)
C$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd,ilist) 
         do ilist=1,nlist
            jbox=list(ilist)
            call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
            level1=box1(1)
            npts=box(15)
            if_use_trunc = 1
            call h3dformta_add_trunc(ier,zk,scale(level1),
     1           sourcesort(1,box(14)),chargesort(box(14)),
     1           npts,center1,
     1           nterms(level1),nterms_eval(1,level1),
     1           rmlexp(iaddr(2,jbox)),wlege,nlege)
         enddo
C$OMP END PARALLEL DO
c
 3251    continue
c
        t2=omp_get_wtime()
        timeinfo(2)=t2-t1
        ifprune_list2 = 0
        call hfmm3d_list2
     1     (zk,bsize,nlev,laddr,scale,nterms,rmlexp,iaddr,epsfmm,
     1     timeinfo,wlists,mptemp,lmptemp,xnodes,wts,nquad,
     1     ifprune_list2)

        t1=omp_get_wtime()
c       ... step 6, adaptive part, evaluate multipole expansions, 
c           or evaluate the potentials and fields directly
         do 3252 ibox=1,nboxes
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         itype=4
         call d3tgetl(ier,ibox,itype,list,nlist,wlists)
c       ... prune all sourceless boxes
         if( box(15) .eq. 0 ) nlist=0
c       ... note that lists 3 and 4 are dual
c       ... evaluate multipole expansions for all boxes in list 4 
c       ... if source is childless, evaluate directly (if cheaper)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1,level,radius)
C$OMP$PRIVATE(ier,i,j,ptemp,ftemp,cd,ilist) 
         do ilist=1,nlist
            jbox=list(ilist)
            call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
            level=box(1)
            call h3dmpevalall_trunc(zk,scale(level),center0,
     1         rmlexp(iaddr(1,ibox)),
     1         nterms(level),nterms_eval(1,level),
     1         sourcesort(1,box1(14)),box1(15),
     1         ifpot,pot(box1(14)),
     1         iffld,fld(1,box1(14)),
     1         wlege,nlege,ier)

        enddo
C$OMP END PARALLEL DO
 3252   continue
        t2=omp_get_wtime()
        timeinfo(6)=t2-t1

        t1=omp_get_wtime()
c       ... step 7, evaluate local expansions and all fields directly
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level,npts,nkids,ier)
        do ibox=1,nboxes
           call d3tgetb(ier,ibox,box,center0,corners0,wlists)
           call d3tnkids(box,nkids)
           if (nkids .eq. 0) then
c     ... evaluate local expansions
              level=box(1)
              npts=box(15)
              if (level .ge. 2) then
                 call h3dtaevalall_trunc(zk,scale(level),center0,
     1                rmlexp(iaddr(2,ibox)),
     1                nterms(level),nterms_eval(1,level),
     1                sourcesort(1,box(14)),box(15),
     1                ifpot,pot(box(14)),
     1                iffld,fld(1,box(14)),
     1                wlege,nlege,ier)
              endif
           endif
        enddo
C$OMP END PARALLEL DO
        t2=omp_get_wtime()
ccc     call prin2('time=*',t2-t1,1)
        timeinfo(7)=t2-t1

        t1=omp_get_wtime()
c       ... step 8, evaluate direct interactions 
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,nkids,list,nlist,npts)
C$OMP$PRIVATE(jbox,box1,center1,corners1)
C$OMP$PRIVATE(ier,ilist,itype) 
C$OMP$SCHEDULE(DYNAMIC)
        do ibox=1,nboxes
           call d3tgetb(ier,ibox,box,center0,corners0,wlists)
           call d3tnkids(box,nkids)
           if (nkids .eq. 0) then
c     ... evaluate self interactions
              call hfmm3dpart_direct_self(zk,box,sourcesort,
     1             chargesort,pot,fld)
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
     1                chargesort,pot,fld)
              enddo
           endif
        enddo
C$OMP END PARALLEL DO
        t2=omp_get_wtime()
        timeinfo(8)=t2-t1
        return
        end

        subroutine hfmm3dpart_direct_self(zk,box,
     1     source,charge,pot,fld)
        implicit real *8 (a-h,o-z)
        integer box(20),box1(20)
        dimension source(3,1)
        complex *16 charge(1),zk
        complex *16 pot(1),fld(3,1)
        complex *16 ptemp,ftemp(3)
        do j=box(14),box(14)+box(15)-1
           do i=box(14),box(14)+box(15)-1
              if (i .eq. j) cycle
              call hpotfld3d(1,source(1,i),charge(i),
     1             source(1,j),zk,ptemp,ftemp)
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
        dimension source(3,1),dipvec(3,1)
        complex *16 charge(1),dipstr(1),zk
        dimension target(3,1)
        complex *16 pot(1),fld(3,1)
        complex *16 ptemp,ftemp(3)
        do j=box1(14),box1(14)+box1(15)-1
        call hpotfld3dall(1,source(1,box(14)),charge(box(14)),
     1     box(15),source(1,j),zk,ptemp,ftemp)
        pot(j)=pot(j)+ptemp
        fld(1,j)=fld(1,j)+ftemp(1)
        fld(2,j)=fld(2,j)+ftemp(2)
        fld(3,j)=fld(3,j)+ftemp(3)
        enddo
        return
        end
