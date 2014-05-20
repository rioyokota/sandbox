************************************************************************
      function next235_cproj(base)
      implicit none
      integer next235_cproj, numdiv
      real*8 base
c ----------------------------------------------------------------------
c     integer function next235_cproj returns a multiple of 2, 3, and 5
c
c     next235_cproj = 2^p 3^q 5^r >= base  where p>=1, q>=0, r>=0
************************************************************************
      next235_cproj = 2 * int(base/2d0+.9999d0)
      if (next235_cproj.le.0) next235_cproj = 2
100   numdiv = next235_cproj
      do while (numdiv/2*2 .eq. numdiv)
         numdiv = numdiv /2
      enddo
      do while (numdiv/3*3 .eq. numdiv)
         numdiv = numdiv /3
      enddo
      do while (numdiv/5*5 .eq. numdiv)
         numdiv = numdiv /5
      enddo
      if (numdiv .eq. 1) return
      next235_cproj = next235_cproj + 2
      goto 100
      end
c***********************************************************************
      subroutine rotviaproj0(beta,nquad,nterms,m1,m2,mpole,lmp,
     1           marray2,lmpn)
c***********************************************************************
C
c       INPUT:
c
c       beta:  the rotation angle about the y-axis.
c       nquad:  number of quadrature points on equator
c       nterms: order of multipole expansion
c
c       m1    : max m index for first expansion   
c               NOT IMPLEMENTED but integer argument must be supplied.
c       m2    : max m index for second expansion
c               NOT IMPLEMENTED but integer argument must be supplied.
c
c       mpole:   coefficients of original multiple expansion
c       lmp:     leading dimension of mpole
c       lmpn:    leading dimension of output array marray2
c
c       OUTPUT:
c
c       marray2  coefficients of rotated expansion.
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nquad,nterms
      real *8 cthetas(nquad),cphis(nquad)
      real *8 sthetas(nquad),sphis(nquad)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      real *8 rat1(0:nterms,0:nterms)
      real *8 rat2(0:nterms,0:nterms)
      complex *16 avec(nquad)
      complex *16 bvec(nquad)
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 marray2(0:lmpn,-lmpn:lmpn)
      complex *16 uder(nquad,0:nterms),uval(nquad,0:nterms)
      complex *16 ephis(-nterms:nterms)
      real *8 wsave(4*nquad+20)
c     Algorithm:
c     1) get locations of quadrature nodes
c     2) evaluate u and du/dtheta
c     3) project onto spherical harmonics.
      call getmeridian(beta,nquad,cthetas,sthetas,cphis,sphis)    
      call evalall2(beta,nquad,cthetas,sthetas,cphis,sphis,mpole,
     2           lmp,nterms,uval,uder,ynm,ynmd,ephis,rat1,rat2)
      call projectonynm2(nquad,uval,uder,ynm,ynmd,marray2,lmpn,nterms,
     2           m2,wsave,avec,bvec,rat1,rat2)
      return
      end
C***********************************************************************
      subroutine getmeridian(beta,nquad,cthetas,sthetas,cphis,sphis)
C***********************************************************************
C     Purpose:
C
C           For a rotation of angle BETA about the y-axis, this
C           subroutine returns the NQUAD equispaced nodes on the 
C           rotated equator in the original coordinate system.
C
C---------------------------------------------------------------------
C     INPUT:
C
C     beta  = angle of rotation
C     nquad = number of quadrature nodes in equator.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     cthetas = cos(theta) values in original coordinate system of 
C                    nquad equispaced nodes
C     sthetas = sin(theta) values in original coordinate system of 
C                    nquad equispaced nodes
C     cphis =  cos(phi) values in original coordinate system of 
C                    nquad equispaced nodes
C     sphis =  cos(phi) values in original coordinate system of 
C                    nquad equispaced nodes
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer nquad
      real *8 cthetas(nquad)
      real *8 sthetas(nquad)
      real *8 cphis(nquad)
      real *8 sphis(nquad)
      pi = 4.0d0*datan(1.0d0)
      ca = cos(beta)
      sa = sin(beta)
      do i = 1,nquad
	 im1 = i-1
         phi = 2*pi*im1/nquad
	 theta = pi/2.0d0
         xp = cos(phi)*sin(theta)
         yp = sin(phi)*sin(theta)
         zp = cos(theta)
         x = ca*xp + sa*zp
         y = yp
         z = -sa*xp + ca*zp
         proj = sqrt(x**2+y**2)
	 if (proj.le.1.0d-16) then
	    cphis(i) = 1.0d0
	    sphis(i) = 0.0d0
	 else
	    cphis(i) = x/proj
	    sphis(i) = y/proj
	 endif
	 cthetas(i) = z
	 sthetas(i) = proj
      enddo
      return
      end
C***********************************************************************
      subroutine evalall0(beta,nquad,cthetas,sthetas,cphis,sphis,
     1           mpole,lmp,nterms,uval,uder,ynm,ynmd,ephis,rat1,rat2)
C***********************************************************************
C
C     This subroutine evaluates the multipole expansion for each
C     order at the nquad nodes on the rotated equator.
C
C---------------------------------------------------------------------
C     INPUT:
C
C     beta    : angle of rotation about y-axis.
C     nquad    : number of target point son unit sphere
C     cthetas  : cos(theta) values of target points.
C     sthetas  : sin(theta) values of target points.
C     cphis    : cos(phi) values of target points.
C     sphis    : sin(phi) values of target points.
C     mpole    : original multipole expansion
C     nterms   : order of multipole expansion
C     ynm      : work array for ynm values
C     ynmd     : work array for ynmd values
C     ephis    : work array for exp(i m phi) values
C     rat1     : work array for accelerating ynm calculation.
C     rat2     : work array for accelerating ynm calculation.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     uval(i,j) : contribution to potential 
C                 of multipole terms of order j at ith quad node.
C     uder(i,j) : contributions to theta derivative of potential
C                 of multipole terms of order j at ith quad node.
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer ndeg,morder, nquad
      real *8 cthetas(nquad),cphis(nquad)
      real *8 sthetas(nquad),sphis(nquad)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 ephi1,ephis(-nterms:nterms)
      complex *16 uder(nquad,0:nterms),uval(nquad,0:nterms)
      complex *16 uv,utheta,uphi,ztmp1,ztmp2,ztsum
      complex *16 ux,uy,uz,imag
      real *8 rat1(0:nterms,0:nterms)
      real *8 rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      pi = 4.0d0*datan(1.0d0)
      cbeta = cos(beta)
      sbeta = -sin(beta)
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquad
	 ctheta = cthetas(jj)
	 stheta = sthetas(jj)
	 cphi = cphis(jj)
	 sphi = sphis(jj)
         dir1 = -sbeta
         dir2 = 0
         dir3 = cbeta
         tang1 = cphi*ctheta
         tang2 = sphi*ctheta
         tang3 = -stheta
         proj2 = tang1*dir1 + tang2*dir2 + tang3*dir3
         tang1 = -sphi
         tang2 = cphi
         tang3 = 0
         proj1 = tang1*dir1 + tang2*dir2 + tang3*dir3
	 call ylgndru2sf(nterms,ctheta,ynm,ynmd,rat1,rat2)
         ephi1 = dcmplx(cphis(jj),sphis(jj))
	 ephis(1) = ephi1
	 ephis(-1) = dconjg(ephi1)
	 do i = 2,nterms
	    ephis(i) = ephis(i-1)*ephi1
	    ephis(-i) = ephis(-i+1)*dconjg(ephi1)
	 enddo
	 do ndeg = 0,nterms
	    uv = ynm(ndeg,0)*mpole(ndeg,0)
	    utheta=ynmd(ndeg,0)*stheta*mpole(ndeg,0)
	    uphi = 0.0d0
	    do morder = 1,ndeg
               ztmp1 = ephis(morder)*mpole(ndeg,morder)
               ztmp2 = ephis(-morder)*mpole(ndeg,-morder)
	       ztsum = ztmp1+ztmp2
	       uv = uv + stheta*ynm(ndeg,morder)*ztsum
	       utheta = utheta + ynmd(ndeg,morder)*ztsum
	       uphi = uphi - ynm(ndeg,morder)*morder*(ztmp1-ztmp2)
	    enddo
            uval(jj,ndeg) = uv
            uder(jj,ndeg) = utheta*proj2+uphi*imag*proj1
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine evalall2(beta,nquad,cthetas,sthetas,cphis,sphis,
     1           mpole,lmp,nterms,uval,uder,ynm,ynmd,ephis,rat1,rat2)
C***********************************************************************
C
C     This subroutine evaluates the multipole expansion for each
C     order at the nquad nodes on the rotated equator.
C
C---------------------------------------------------------------------
C     INPUT:
C
C     beta    : angle of rotation about y-axis.
C     nquad    : number of target point son unit sphere
C     cthetas  : cos(theta) values of target points.
C     sthetas  : sin(theta) values of target points.
C     cphis    : cos(phi) values of target points.
C     sphis    : sin(phi) values of target points.
C     mpole    : original multipole expansion
C     nterms   : order of multipole expansion
C     ynm      : work array for ynm values
C     ynmd     : work array for ynmd values
C     ephis    : work array for exp(i m phi) values
C     rat1     : work array for accelerating ynm calculation.
C     rat2     : work array for accelerating ynm calculation.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     uval(i,j) : contribution to potential 
C                 of multipole terms of order j at ith quad node.
C     uder(i,j) : contributions to theta derivative of potential
C                 of multipole terms of order j at ith quad node.
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer ndeg,morder, nquad
      real *8 cthetas(nquad),cphis(nquad)
      real *8 sthetas(nquad),sphis(nquad)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 ephi1,ephis(-nterms:nterms)
      complex *16 uder(nquad,0:nterms),uval(nquad,0:nterms)
      complex *16 uv,utheta,uphi,ztmp1,ztmp2,ztsum,ztdif
      complex *16 ux,uy,uz,ima
      real *8 rat1(0:nterms,0:nterms)
      real *8 rat2(0:nterms,0:nterms)
      data ima/(0.0d0,1.0d0)/
      pi = 4.0d0*datan(1.0d0)
      cbeta = cos(beta)
      sbeta = -sin(beta)
      call ylgndrini(nterms,rat1,rat2)
      nquad2=nquad/2
      if( mod(nquad2,2) .eq. 0 ) nquad4=nquad2/2+1
      if( mod(nquad2,2) .eq. 1 ) nquad4=nquad2/2+1
      do jj=1,nquad4
	 ctheta = cthetas(jj)
	 stheta = sthetas(jj)
	 cphi = cphis(jj)
	 sphi = sphis(jj)
         dir1 = -sbeta
         dir2 = 0
         dir3 = cbeta
         tang1 = cphi*ctheta
         tang2 = sphi*ctheta
         tang3 = -stheta
         proj2 = tang1*dir1 + tang2*dir2 + tang3*dir3
         tang1 = -sphi
         tang2 = cphi
         tang3 = 0
         proj1 = tang1*dir1 + tang2*dir2 + tang3*dir3
	 call ylgndru2sf(nterms,ctheta,ynm,ynmd,rat1,rat2)
         ephi1 = dcmplx(cphis(jj),sphis(jj))
	 ephis(1) = ephi1
	 ephis(-1) = dconjg(ephi1)
	 do i = 2,nterms
	    ephis(i) = ephis(i-1)*ephi1
	    ephis(-i) = dconjg(ephis(i))
	 enddo
	 do ndeg = 0,nterms
	    uv=0
	    utheta=0
	    uphi=0
	    do morder = 1,ndeg
               ztmp1=ephis(morder)*mpole(ndeg,morder)
               ztmp2=ephis(-morder)*mpole(ndeg,-morder)
	       ztsum=ztmp1+ztmp2
	       ztdif=ztmp1-ztmp2
	       uv=uv+ynm(ndeg,morder)*ztsum
	       utheta=utheta+ynmd(ndeg,morder)*ztsum
	       uphi=uphi-ynm(ndeg,morder)*morder*ztdif
	    enddo
	    uv=stheta*uv+ynm(ndeg,0)*mpole(ndeg,0)
	    utheta=utheta+ynmd(ndeg,0)*stheta*mpole(ndeg,0)
c       ... apply the periodizing operator
            uval(jj,ndeg) = uv
            uder(jj,ndeg) = (utheta*proj2+uphi*ima*proj1)
            if( mod(ndeg,2) .eq. 0 ) then
            uval(jj+nquad/2,ndeg) = +uval(jj,ndeg)
            uder(jj+nquad/2,ndeg) = -uder(jj,ndeg)
            endif
            if( mod(ndeg,2) .eq. 1 ) then
            uval(jj+nquad/2,ndeg) = -uval(jj,ndeg)
            uder(jj+nquad/2,ndeg) = +uder(jj,ndeg)
            endif
c       ... alternative form of the periodizing operator
	 enddo

         if_reflect=1
         if( jj .eq. 1 ) if_reflect=0
         if( jj .eq. nquad4 .and. mod(nquad2,2) .eq. 0 ) if_reflect=0
         if( if_reflect .eq. 1 ) then
         jjr=nquad2 - jj + 2
         call ylgndr2pm_opt(nterms,ynm,ynmd)

	 ctheta = -cthetas(jj)
	 stheta = sthetas(jj)
	 cphi = -cphis(jj)
	 sphi = sphis(jj)
         dir1 = -sbeta
         dir2 = 0
         dir3 = cbeta
         tang1 = cphi*ctheta
         tang2 = sphi*ctheta
         tang3 = -stheta
         proj2 = tang1*dir1 + tang2*dir2 + tang3*dir3
         tang1 = -sphi
         tang2 = cphi
         tang3 = 0
         proj1 = tang1*dir1 + tang2*dir2 + tang3*dir3

         ephi1 = dcmplx(cphi,sphi)
	 ephis(1) = ephi1
	 ephis(-1) = dconjg(ephi1)
	 do i = 2,nterms
	    ephis(i) = ephis(i-1)*ephi1
	    ephis(-i) = dconjg(ephis(i))
	 enddo
	 do ndeg = 0,nterms
	    uv=0
	    utheta=0
	    uphi=0
	    do morder = 1,ndeg
               ztmp1=ephis(morder)*mpole(ndeg,morder)
               ztmp2=ephis(-morder)*mpole(ndeg,-morder)
	       ztsum=ztmp1+ztmp2
	       ztdif=ztmp1-ztmp2
	       uv=uv+ynm(ndeg,morder)*ztsum
	       utheta=utheta+ynmd(ndeg,morder)*ztsum
	       uphi=uphi-ynm(ndeg,morder)*morder*ztdif
	    enddo
	    uv=stheta*uv+ynm(ndeg,0)*mpole(ndeg,0)
	    utheta=utheta+ynmd(ndeg,0)*stheta*mpole(ndeg,0)
c       ... apply the periodizing operator
            uval(jjr,ndeg) = uv
            uder(jjr,ndeg) = (utheta*proj2+uphi*ima*proj1)
            if( mod(ndeg,2) .eq. 0 ) then
            uval(jjr+nquad/2,ndeg) = +uval(jjr,ndeg)
            uder(jjr+nquad/2,ndeg) = -uder(jjr,ndeg)
            endif
            if( mod(ndeg,2) .eq. 1 ) then
            uval(jjr+nquad/2,ndeg) = -uval(jjr,ndeg)
            uder(jjr+nquad/2,ndeg) = +uder(jjr,ndeg)
            endif
c       ... alternative form of the periodizing operator
	 enddo
         endif
      enddo
      return
      end
C***********************************************************************
      subroutine projectonynm2(nquad,uval,uder,
     1           ynm,ynmd,marray,lmpn,nterms,m2,wsave,avec,bvec,
     $           rat1,rat2)
C***********************************************************************
C
C     This subroutine projects from values on equator for each multipole
C     order (uval, uder = dudthteta) 
C     onto spherical harmonics
C
C---------------------------------------------------------------------
C     INPUT:
C
C     nquad    : number of points on equator
C     uval     : F values on equator
C     uder     : dFdtheta values on equator
C     ynm      : work array for ynm values
C     ynmd     : work array for ynmd values
C     lmpn     : leading dim of marray (must exceed nterms)
C     nterms   : order of expansion
C     m2       : NOT IMPLEMENTED (for reduced number of degrees in 
C                expansion (second index)
C     wsave    : work array for FFT (dimension at least 4*nquad+20)
C     avec     : work array of length nquad for FFT (complex)
C     bvec     : work array of length nquad for FFT (complex)
C---------------------------------------------------------------------
C     OUTPUT:
C
C     marray   : rotated expansion 
C
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer nquad, norder
      complex *16 ephi,ephi1,uval(nquad,0:1)
      complex *16 uder(nquad,0:1)
      complex *16 utheta,uphi,ztmp1,ztmp2
      complex *16 alpha,beta,ima
      complex *16 marray(0:lmpn,-lmpn:lmpn)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      real *8 wsave(4*nquad+20)
      complex *16 avec(nquad)
      complex *16 bvec(nquad)
      real *8 rat1(0:nterms,0:nterms)
      real *8 rat2(0:nterms,0:nterms)
      data ima/(0.0d0,1.0d0)/
      ctheta = 0.0d0
      stheta = 1.0d0
      h = 1.0d0/nquad
      call ylgndru2sf(nterms,ctheta,ynm,ynmd,rat1,rat2)
      call zffti(nquad,wsave)
      do norder=0,nterms
         d=sqrt(2*norder+1.0d0)
	 do ii = 1,nquad
	    avec(ii) = uval(ii,norder)*d + uder(ii,norder)
         enddo
	 call zfftf(nquad,avec,wsave)
         do m = -norder,norder
	    if (m.ge.0)  alpha = avec(m+1)*h
	    if (m.lt.0)  alpha = avec(nquad+m+1)*h
            marray(norder,m) = alpha/
     1        (ynm(norder,abs(m))*d - (ynmd(norder,abs(m))))
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine rotviaprojf90(beta,nterms,mpole,lmp,
     1           marray2,lmpn)
c***********************************************************************
c       Purpose:
c
c	Fast and stable algorithm for applying rotation operator about
c	the y-axis determined by angle beta.
c
c       The method is based on computing the induced potential and
c       its theta-derivative on the rotated equator
c       for each order (first index). The coefficients of  the rotated
c       expansion can then be obtained by FFT and projection.
c
c       There is some loss in speed over using recurrence relations 
c       but it is stable to all orders whereas the recurrence schemes 
c       are not.
c
c       If the rotation operator is to be used multiple times, and
c       memory is available, one can precompute and store the 
c       multipliers used in evalall (see below). This has not yet been
c       implemented.
c
C---------------------------------------------------------------------
c       INPUT:
c
c       beta:  the rotation angle about the y-axis.
c       nterms: order of multipole expansion
C       mpole   coefficients of original multiple expansion
C       lmp     leading dim for mpole (must exceed nterms)
C       lmpn    leading dim for marray2 (must exceed nterms)
c
C---------------------------------------------------------------------
c       OUTPUT:
c
c       marray2  coefficients of rotated expansion.
c
C---------------------------------------------------------------------
      implicit none
      integer nquad,ier,nterms,lmp,lmpn,next235_cproj
      real *8 beta
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 marray2(0:lmpn,-lmpn:lmpn)
      nquad = next235_cproj((2*nterms+2)*1.0d0)
      call rotviaproj0(beta,nquad,nterms,nterms,nterms,mpole,lmp,
     1           marray2,lmpn)
      return
      end

      subroutine rotate(theta,ntermsj,Mnm,ntermsi,Mrot)
      implicit none
      integer ntermsj,ntermsi
      real *8 theta
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsi,-ntermsi:ntermsi)
      if( ntermsj .ge. 30 ) then
         call rotviaprojf90(theta,ntermsj,Mnm,ntermsj,Mrot,ntermsi)
      else
         call rotviarecur3f90(theta,ntermsj,Mnm,ntermsj,Mrot,ntermsi)
      endif
      return
      end
