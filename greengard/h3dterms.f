      subroutine h3dterms(size, wavek, eps, nterms, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter wavek.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  wavek, z1, z2, z3, jfun(0:2000), ht0,
     1             ht1, ht2, fjder(0:1), ztmp,
     1             hfun(0:2000), fhder(0:1)
      ier = 0
      z1 = (wavek*size)*1.5d0
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(wavek*size) .lt. 1.0d0) rscale = cdabs(wavek*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
      z2 = (wavek*size) * dsqrt(3d0)/2.d0
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


      subroutine h3dterms_list2(size, wavek, eps, itable, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter wavek.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Build nterms table for all boxes in list 2
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  wavek, z1, z2, z3, jfun(0:2000), ht0,
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
        z1 = (wavek*size)*rr
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(wavek*size) .lt. 1.0d0) rscale = cdabs(wavek*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
      z2 = (wavek*size) * dsqrt(3d0)/2.d0
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

      subroutine h3dterms_eval(itype, size, wavek, eps, nterms, ier)
      implicit real *8 (a-h,o-z)
c     Determine number of terms in mpole expansions for box of size
c     "size" with Helmholtz parameter wavek.
c
c     The method is based on examining the decay of h_n * j_n.
c
c     Maximum number of terms is 1000, which 
c     works for boxes up to 160 wavelengths in size     
      integer iscale(0:2000)
      complex *16  wavek, z1, z2, z3, jfun(0:2000), ht0,
     1             ht1, ht2, fjder(0:1), ztmp,
     1             hfun(0:2000), fhder(0:1)
      ier = 0
      z1 = (wavek*size)*1.5d0
c       the code will run out memory if frequency is too small 
c       set frequency to something more reasonable, nterms is 
c       approximately the same for all small frequencies
      ntmax = 1000
      ifder = 0
      rscale = 1.0d0
      if (cdabs(wavek*size) .lt. 1.0d0) rscale = cdabs(wavek*size)
      call h3dall(ntmax,z1,rscale,hfun,ifder,fhder)
        z2 = (wavek*size) * dsqrt(3d0)/2.d0
c       corners included
        if( itype .eq. 1 ) z2 = (wavek*size) * dsqrt(3d0)/2.d0
c       edges included, no corners
        if( itype .eq. 2 ) z2 = (wavek*size) * dsqrt(2d0)/2.d0
c       center only
        if( itype .eq. 3 ) z2 = (wavek*size) * 1.0d0/2.d0
c       center only, small interior sphere
        if( itype .eq. 4 ) z2 = (wavek*size) * 0.8d0/2.d0
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

c**********************************************************************
      subroutine h3dall(nterms,z,scale,hvec,ifder,hder)
c**********************************************************************
c     This subroutine computes scaled versions of the spherical Hankel 
c     functions h_n of orders 0 to nterms.
c
c       	hvec(n)= h_n(z)*scale^(n)
c
c     The parameter SCALE is useful when |z| < 1, in which case
c     it damps out the rapid growth of h_n as n increases. In such 
c     cases, we recommend setting 
c                                 
c               scale = |z|
c
c     or something close. If |z| > 1, set scale = 1.
c
c     If the flag IFDER is set to one, it also computes the 
c     derivatives of h_n.
c
c		hder(n)= h_n'(z)*scale^(n)
c
c     NOTE: If |z| < 1.0d-15, the subroutine returns zero.
c-----------------------------------------------------------------------
c     INPUT:
c     nterms  : highest order of the Hankel functions to be computed.
c     z       : argument of the Hankel functions.
c     scale   : scaling parameter discussed above
c     ifder   : flag indcating whether derivatives should be computed.
c		ifder = 1   ==> compute 
c		ifder = 0   ==> do not compute
c-----------------------------------------------------------------------
c     OUTPUT:
c     hvec    : the vector of spherical Hankel functions 
c     hder    : the derivatives of the spherical Hankel functions 
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      complex *16 hvec(0:nterms),hder(0:nterms)
      complex *16 wavek2,z,zinv,ztmp,fhextra
      data thresh/1.0d-15/,done/1.0d0/
c     If |z| < thresh, return zeros.
      if (abs(z).lt.thresh) then
         do i=0,nterms
            hvec(i)=0
            hder(i)=0
         enddo
         return
      endif
c     Otherwise, get h_0 and h_1 analytically and the rest via recursion.
      call h3d01(z,hvec(0),hvec(1))
      hvec(0)=hvec(0)
      hvec(1)=hvec(1)*scale
c     From Abramowitz and Stegun (10.1.19)
c     h_{n+1}(z)=(2n+1)/z * h_n(z) - h_{n-1}(z)
c     With scaling:
c     hvec(n+1)=scale*(2n+1)/z * hvec(n) -(scale**2) hvec(n-1)
      scal2=scale*scale
      zinv=scale/z
      do i=1,nterms-1
	 dtmp=(2*i+done)
	 ztmp=zinv*dtmp
	 hvec(i+1)=ztmp*hvec(i)-scal2*hvec(i-1)
      enddo
c     From Abramowitz and Stegun (10.1.21)
c     h_{n}'(z)= h_{n-1}(z) - (n+1)/z * h_n(z)
c     With scaling:
c     hder(n)=scale* hvec(n-1) - (n+1)/z * hvec(n)
      if (ifder.eq.1) then
	 hder(0)=-hvec(1)/scale
         zinv=1.0d0/z
         do i=1,nterms
	    dtmp=(i+done)
	    ztmp=zinv*dtmp
	    hder(i)=scale*hvec(i-1)-ztmp*hvec(i)
	 enddo
      endif
      return
      end
c**********************************************************************
      subroutine h3d01(z,h0,h1)
c**********************************************************************
c     Compute spherical Hankel functions of order 0 and 1 
c     h0(z)  =   exp(i*z)/(i*z),
c     h1(z)  =   - h0' = -h0*(i-1/z) = h0*(1/z-i)
c-----------------------------------------------------------------------
c     INPUT:
c	z   :  argument of Hankel functions
c              if abs(z)<1.0d-15, returns zero.
c-----------------------------------------------------------------------
c     OUTPUT:
c	h0  :  h0(z)    (spherical Hankel function of order 0).
c	h1  :  -h0'(z)  (spherical Hankel function of order 1).
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      complex *16 z,zinv,eye,cd,h0,h1
      data eye/(0.0d0,1.0d0)/, thresh/1.0d-15/, done/1.0d0/
      if (abs(z).lt.thresh) then
         h0=0.0d0
         h1=0.0d0
         return
      endif
      cd = eye*z
      h0=exp(cd)/cd
      h1=h0*(done/z - eye)
      return
      end
