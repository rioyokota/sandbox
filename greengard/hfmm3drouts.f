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
        dimension nterms(0:200)
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
