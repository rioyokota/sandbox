      subroutine h3dterms(size, zk, eps, nterms, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter zk.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  zk, z1, z2, z3, jfun(0:2000), ht0,
     1             ht1, ht2, fjder(0:1), ztmp,
     1             hfun(0:2000), fhder(0:1)
      ier = 0
      z1 = (zk*size)*1.5d0
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(zk*size) .lt. 1.0d0) rscale = cdabs(zk*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
      z2 = (zk*size) * dsqrt(3d0)/2.d0
      ier1 = 0
      call jfuns3d(ier1, ntmax, z2, rscale, jfun, ifder, fjder,
     1                 2000, iscale, ntop)
      if (ier1.eq.8) then 
        ier = 11 
        return
      endif        
      xtemp1 = cdabs(jfun(0)*hfun(0))
      xtemp2 = cdabs(jfun(1)*hfun(1))
      xtemp0 = xtemp1+xtemp2
      nterms = 1
      do j = 2, ntmax
        xtemp1 = cdabs(jfun(j)*hfun(j))
        xtemp2 = cdabs(jfun(j-1)*hfun(j-1))
        xtemp = xtemp1+xtemp2
        if(xtemp .lt. eps*xtemp0)then
          nterms = j + 1
          return
        endif
      enddo
c       ... computational box is too big, set nterms to 1000
        ier = 13
        nterms=1000
      return
      end


      subroutine h3dterms_list2(size, zk, eps, itable, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter zk.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Build nterms table for all boxes in list 2
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  zk, z1, z2, z3, jfun(0:2000), ht0,
     1             ht1, ht2, fjder(0:1), ztmp,
     1             hfun(0:2000), fhder(0:1)
      dimension nterms_table(2:3,0:3,0:3)
      dimension itable(-3:3,-3:3,-3:3)
      ier = 0
        do 1800 ii=2,3
        do 1800 jj=0,3
        do 1800 kk=0,3
        dx=ii
        dy=jj
        dz=kk
        if( dx .gt. 0 ) dx=dx-.5
        if( dy .gt. 0 ) dy=dy-.5
        if( dz .gt. 0 ) dz=dz-.5
        rr=sqrt(dx*dx+dy*dy+dz*dz)
        z1 = (zk*size)*rr
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(zk*size) .lt. 1.0d0) rscale = cdabs(zk*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
      z2 = (zk*size) * dsqrt(3d0)/2.d0
      ier1 = 0
      call jfuns3d(ier1, ntmax, z2, rscale, jfun, ifder, fjder,
     1                 2000, iscale, ntop)
      if (ier1.eq.8) then 
        ier = 11 
        return
      endif        
      xtemp1 = cdabs(jfun(0)*hfun(0))
      xtemp2 = cdabs(jfun(1)*hfun(1))
      xtemp0 = xtemp1+xtemp2
      nterms = 1
      do j = 2, ntmax
        xtemp1 = cdabs(jfun(j)*hfun(j))
        xtemp2 = cdabs(jfun(j-1)*hfun(j-1))
        xtemp = xtemp1+xtemp2
        if(xtemp .lt. eps*xtemp0)then
          nterms = j + 1
          goto 1600
        endif
      enddo
c       ... computational box is too big, set nterms to 1000
        ier = 13
        nterms=1000
 1600   continue
        nterms_table(ii,jj,kk)=nterms
 1800   continue
c       build the rank table for all boxes in list 2
        do i=-3,3
        do j=-3,3
        do k=-3,3
        itable(i,j,k)=0
        enddo
        enddo
        enddo
        do 2200 k=-3,3
        do 2200 i=-3,3
        do 2200 j=-3,3
        if( abs(i) .gt. 1 ) then
        itable(i,j,k)=nterms_table(abs(i),abs(j),abs(k))
        else if( abs(j) .gt. 1) then
        itable(i,j,k)=nterms_table(abs(j),abs(i),abs(k))
        endif
        if( abs(i) .le. 1 .and. abs(j) .le. 1) then
        if( abs(k) .gt. 1 ) then
        if( abs(i) .ge. abs(j) ) then
        itable(i,j,k)=nterms_table(abs(k),abs(i),abs(j))
        else
        itable(i,j,k)=nterms_table(abs(k),abs(j),abs(i))
        endif
        endif
        endif
 2200   continue
      return
      end

      subroutine h3dterms_eval(itype, size, zk, eps, nterms, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter zk.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  zk, z1, z2, z3, jfun(0:2000), ht0,
     1             ht1, ht2, fjder(0:1), ztmp,
     1             hfun(0:2000), fhder(0:1)
      ier = 0
      z1 = (zk*size)*1.5d0
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(zk*size) .lt. 1.0d0) rscale = cdabs(zk*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
        z2 = (zk*size) * dsqrt(3d0)/2.d0
c       corners included
        if( itype .eq. 1 ) z2 = (zk*size) * dsqrt(3d0)/2.d0
c       edges included, no corners
        if( itype .eq. 2 ) z2 = (zk*size) * dsqrt(2d0)/2.d0
c       center only
        if( itype .eq. 3 ) z2 = (zk*size) * 1.0d0/2.d0
c       center only, small interior sphere
        if( itype .eq. 4 ) z2 = (zk*size) * 0.8d0/2.d0
      ier1 = 0
      call jfuns3d(ier1, ntmax, z2, rscale, jfun, ifder, fjder,
     1                 2000, iscale, ntop)
c     set error flag if jfuns runs out of memory
      if (ier1.eq.8) then 
        ier = 11 
        return
      endif        
      xtemp1 = cdabs(jfun(0)*hfun(0))
      xtemp2 = cdabs(jfun(1)*hfun(1))
      xtemp0 = xtemp1+xtemp2
      nterms = 1
      do j = 2, ntmax
        xtemp1 = cdabs(jfun(j)*hfun(j))
        xtemp2 = cdabs(jfun(j-1)*hfun(j-1))
        xtemp = xtemp1+xtemp2
        if(xtemp .lt. eps*xtemp0)then
          nterms = j + 1
          return
        endif
      enddo
c       ... computational box is too big, set nterms to 1000
        ier = 13
        nterms=1000
      return
      end
