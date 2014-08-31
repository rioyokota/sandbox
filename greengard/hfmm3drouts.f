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
        dimension iaddr(nboxes)
        dimension center0(3),corners0(3,8)
        dimension wlists(1)
        iptr=1
        do ibox=1,nboxes
        call d3tgetb(ier,ibox,box,center0,corners0,wlists)
        level=box(1)
        iaddr(ibox)=iptr
        iptr=iptr+(nterms(level)+1)*(2*nterms(level)+1)*2
        enddo
        lmptot = iptr
        return
        end
