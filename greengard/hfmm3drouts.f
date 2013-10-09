        subroutine hfmm3dparttree(ier,iprec,zk,
     $     nsource,source,
     $     nbox,epsfmm,iisource,iwlists,lwlists,
     $     nboxes,laddr,nlev,center,size,
     $     w,lw,lused7)
        implicit real *8 (a-h,o-z)
        dimension source(3,1)
        dimension center(3)
        dimension laddr(2,200)
        integer box(20)
        dimension center0(3),corners0(3,8)
        integer box1(20)
        dimension center1(3),corners1(3,8)
        complex *16 zk
        dimension w(1)
        ier=0
        done=1
        pi=4*atan(done)
        lused7=0
        ifprint=0
        iisource=1
        lused7=lused7+nsource
        if (lused7 .ge. lw) ier=128
        if( ier .ne. 0 ) return
        if (lused7 .ge. lw) ier=128
        if( ier .ne. 0 ) return
        iwlists=iisource+lused7+10
c       ... construct the adaptive FMM oct-tree structure
        call d3tstrcr(ier,source,nsource,nbox,
     $     nboxes,w(iisource),laddr,nlev,center,size,
     $     w(iwlists),lw-lused7,lused)
        if( ier .ne. 0 ) return
        lwlists=lused
        lused7=lused7+lwlists
        if (lused7 .ge. lw) ier=128
        if( ier .ne. 0 ) return
        return
        end

        subroutine hfmm3d_list2
     $     (zk,bsize,nlev,laddr,scale,nterms,rmlexp,iaddr,epsfmm,
     $     timeinfo,wlists,mptemp,lmptemp,xnodes,wts,nquad,
     $     ifprune_list2)
        implicit real *8 (a-h,o-z)
        integer iaddr(2,1),laddr(2,1),nterms(0:1)
        dimension rmlexp(1),scale(0:1),itable(-3:3,-3:3,-3:3)
        integer list(10 000)
        integer box(20)
        dimension bsize(0:200)
        dimension xnodes(nquad)
        dimension wts(nquad)
        dimension center0(3),corners0(3,8)
        integer box1(20)
        dimension center1(3),corners1(3,8)
        dimension wlists(1)
        complex *16 mptemp(lmptemp)
        complex *16 zk
        complex *16 pot(1),fld(3,1),pottarg(1),fldtarg(3,1)
        complex *16 ptemp,ftemp(3)
        real *8, allocatable :: rotmatf(:,:,:,:)
        real *8, allocatable :: rotmatb(:,:,:,:)
        real *8, allocatable :: thetas(:,:,:)
        real *8 rvec(3)
        dimension timeinfo(10)
        real *8, allocatable :: xnodes2(:), wts2(:)
        max_nodes = 10000
        allocate( xnodes2(max_nodes) )
        allocate( wts2(max_nodes) )
        t1=omp_get_wtime()
c       ... step 3, merge all multipole expansions
         do 2300 ilev=nlev,3,-1
        nquad2=nterms(ilev-1)*2.5
        nquad2=max(6,nquad2)
        ifinit2=1
        call legewhts(nquad2,xnodes2,wts2,ifinit2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level0,level,npts,nkids,radius)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1)
C$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd) 
         do 2200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
         call d3tgetb(ier,ibox,box,center0,corners0,wlists)
         call d3tnkids(box,nkids)
c       ... prune all sourceless boxes
         if( box(15) .eq. 0 ) goto 2200
         if (nkids .ne. 0) then
         level0=box(1)
         if( level0 .ge. 2 ) then
            radius = (corners0(1,1) - center0(1))**2
            radius = radius + (corners0(2,1) - center0(2))**2
            radius = radius + (corners0(3,1) - center0(3))**2
            radius = sqrt(radius)
            if( ifprint .ge. 2 ) then
               call prin2('radius=*',radius,1)
               call prinf('ibox=*',ibox,1)
               call prinf('box=*',box,20)
               call prinf('nkids=*',nkids,1)
            endif
c       ... merge multipole expansions of the kids 
            call h3dzero(rmlexp(iaddr(1,ibox)),nterms(level0))
            if (ifprint .ge. 2) then
               call prin2('center0=*',center0,3)
            endif
            do 2100 i = 1,8
               jbox = box(5+i)
               if (jbox.eq.0) goto 2100
               call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
               if (ifprint .ge. 2) then
               call prinf('jbox=*',jbox,1)
               call prin2('center1=*',center1,3)
               endif
               level1=box1(1)
               call h3dmpmpquadu_add(zk,scale(level1),center1,
     1            rmlexp(iaddr(1,jbox)),nterms(level1),scale(level0),
     1            center0,rmlexp(iaddr(1,ibox)),
     $            nterms(level0),nterms(level0),
     1            radius,xnodes2,wts2,nquad2,ier)
 2100       continue
            if (ifprint .ge. 2) then
            call prinf('=============*',x,0)
            endif
c       ... mark the local expansion of all kids and the parent
            endif
         endif
 2200    continue
C$OMP END PARALLEL DO
 2300    continue
        t2=omp_get_wtime()
        timeinfo(3)=t2-t1

        t1=omp_get_wtime()
c       ... precompute rotation matrices, useful up to order 10 or so
c       (approximately 30kB of storage for ldm=10)
c       (approximately 40MB of storage for ldm=30)
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
        call cart2polar(rvec,d,theta,phi)
        thetas(i,j,k)=theta
        call rotviarecur3p_init(ier,rotmatf(1,i,j,k),ldm,+theta)
        call rotviarecur3p_init(ier,rotmatb(1,i,j,k),ldm,-theta)
        endif
        enddo
        enddo
        enddo
c       ... step 4, convert multipole expansions into the local ones
        do 4300 ilev=3,nlev+1
        call h3dterms_list2(bsize(ilev-1),zk,epsfmm, itable, ier)
        nquad2=nterms(ilev-1)*1.2
        nquad2=max(6,nquad2)
        ifinit2=1
        call legewhts(nquad2,xnodes2,wts2,ifinit2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,list,nlist)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1,ifdirect2,radius)
C$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd,ilist,itype)
C$OMP$PRIVATE(if_use_trunc,nterms_trunc,ii,jj,kk) 
C$OMP$SCHEDULE(DYNAMIC)
        do 4200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        level0=box(1)
        if (level0 .ge. 2) then
c       ... retrieve list #2
           itype=2
           call d3tgetl(ier,ibox,itype,list,nlist,wlists)
c       ... for all pairs in list #2, apply the translation operator 
           do 4150 ilist=1,nlist
              jbox=list(ilist)
              call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
              if (box1(15).eq.0) goto 4150
              if ((box(17).eq.0).and.(ifprune_list2.eq.1))
     $           goto 4150
              radius = (corners1(1,1) - center1(1))**2
              radius = radius + (corners1(2,1) - center1(2))**2
              radius = radius + (corners1(3,1) - center1(3))**2
              radius = sqrt(radius)
c       ... convert multipole expansions for all boxes in list 2 in local exp
c       ... if source is childless, evaluate directly (if cheaper)
              level1=box1(1)
              ifdirect2 = 0
              ii=box1(2)-box(2)
              jj=box1(3)-box(3)
              kk=box1(4)-box(4)
              nterms_trunc=itable(ii,jj,kk)
              nterms_trunc=min(nterms(level0),nterms_trunc)
              nterms_trunc=min(nterms(level1),nterms_trunc)
              if (ifdirect2 .eq. 0) then
              if_use_rotmatfb = 1
              if( nterms(level0) .gt. ldm ) if_use_rotmatfb = 0
              if( nterms(level1) .gt. ldm ) if_use_rotmatfb = 0
              if( nterms_trunc   .gt. ldm ) if_use_rotmatfb = 0
              if( if_use_rotmatfb .eq. 1 ) then
              call h3dmplocquadu2_add_trunc(zk,scale(level1),center1,
     1           rmlexp(iaddr(1,jbox)),nterms(level1),nterms_trunc,
     $           scale(level0),center0,
     $           rmlexp(iaddr(2,ibox)),nterms(level0),nterms_trunc,
     2           radius,xnodes2,wts2,nquad2,ier,
     $           rotmatf(1,-ii,-jj,-kk),rotmatb(1,-ii,-jj,-kk),ldm)
              else
              call h3dmplocquadu_add_trunc(zk,scale(level1),center1,
     1           rmlexp(iaddr(1,jbox)),nterms(level1),nterms_trunc,
     $           scale(level0),center0,
     $           rmlexp(iaddr(2,ibox)),nterms(level0),nterms_trunc,
     2           radius,xnodes2,wts2,nquad2,ier)
              endif
              endif
 4150       continue
        endif
 4200   continue
 4300   continue
c
        t2=omp_get_wtime()
        timeinfo(4)=t2-t1

        t1=omp_get_wtime()
c       ... step 5, split all local expansions
        do 5300 ilev=3,nlev
        nquad2=nterms(ilev-1)*2
        nquad2=max(6,nquad2)
        ifinit2=1
        call legewhts(nquad2,xnodes2,wts2,ifinit2)
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(ibox,box,center0,corners0,level0,level,npts,nkids,radius)
C$OMP$PRIVATE(jbox,box1,center1,corners1,level1)
C$OMP$PRIVATE(lused,ier,i,j,ptemp,ftemp,cd) 
        do 5200 ibox=laddr(1,ilev),laddr(1,ilev)+laddr(2,ilev)-1
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        call d3tnkids(box,nkids)
        if (nkids .ne. 0) then
            level0=box(1)
            if (level0 .ge. 2) then
c       ... split local expansion of the parent box
               do 5100 i = 1,8
	          jbox = box(5+i)
	          if (jbox.eq.0) goto 5100
                  call d3tgetb(ier,jbox,box1,center1,corners1,wlists)
                  radius = (corners1(1,1) - center1(1))**2
                  radius = radius + (corners1(2,1) - center1(2))**2
                  radius = radius + (corners1(3,1) - center1(3))**2
                  radius = sqrt(radius)
                  level1=box1(1)
	          call h3dloclocquadu_add(zk,scale(level0),center0,
     1    	      rmlexp(iaddr(2,ibox)),nterms(level0),
     1                scale(level1),center1,rmlexp(iaddr(2,jbox)),
     $                nterms(level1),nterms(level1),
     1    	      radius,xnodes2,wts2,nquad2,ier)
 5100          continue
            endif
        endif
 5200   continue
C$OMP END PARALLEL DO
 5300   continue
        t2=omp_get_wtime()
        timeinfo(5)=t2-t1
        return
        end

        subroutine h3dpsort(n,isource,psort,pot)
        implicit real *8 (a-h,o-z)
        dimension isource(1)
        complex *16 pot(1),psort(1)
        do i=1,n
           pot(isource(i))=psort(i)
        enddo
        return
        end

        subroutine h3dfsort(n,isource,fldsort,fld)
        implicit real *8 (a-h,o-z)
        dimension isource(1)
        complex *16 fld(3,1),fldsort(3,1)
        do i=1,n
           fld(1,isource(i))=fldsort(1,i)
           fld(2,isource(i))=fldsort(2,i)
           fld(3,isource(i))=fldsort(3,i)
        enddo
        return
        end

        subroutine h3dreorder(nsource,source,
     $     ifcharge,charge,isource,sourcesort,chargesort) 
        implicit real *8 (a-h,o-z)
        dimension source(3,1),sourcesort(3,1),isource(1)
        complex *16 charge(1),chargesort(1)
        do i = 1,nsource
           sourcesort(1,i) = source(1,isource(i))
           sourcesort(2,i) = source(2,isource(i))
           sourcesort(3,i) = source(3,isource(i))
           chargesort(i) = charge(isource(i))
        enddo
        return
        end

        subroutine h3dzero(mpole,nterms)
        implicit real *8 (a-h,o-z)
        complex *16 mpole(0:nterms,-nterms:nterms)
        do n=0,nterms
        do m=-n,n
        mpole(n,m)=0
        enddo
        enddo
        return
        end

        subroutine h3dmpalloc(wlists,iaddr,nboxes,lmptot,nterms)
        implicit real *8 (a-h,o-z)
        integer box(20)
        dimension nterms(0:1)
        dimension iaddr(2,nboxes)
        dimension center0(3),corners0(3,8)
        dimension wlists(1)
        iptr=1
        do ibox=1,nboxes
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        level=box(1)
c       ... first, allocate memory for the multipole expansion
        iaddr(1,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*(2*nterms(level)+1)*2
c       ... then, allocate memory for the local expansion
        iaddr(2,ibox)=iptr
        iptr=iptr+(nterms(level)+1)*(2*nterms(level)+1)*2
        enddo
        lmptot = iptr
        return
        end
