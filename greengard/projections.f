C***********************************************************************
      subroutine h3drescalemp(nterms,lmp,mpole,
     1           radius,wavek0,scale)
C***********************************************************************
C
C     This subroutine rescales a spherical harmonic expansion
C     on the surface by h_n(wavek0 * radius), converting 
C     a surface function to the corresponding Hankel expansion
C     in the exterior.
C
C---------------------------------------------------------------------
C     INPUT:
C
C           nterms = order of spherical harmonic expansion
C           mpole = coefficients of s.h. expansion
C           radius = sphere radius
C           wavek0 = Helmholtz parameter
C           scale = scale parameter for expansions
C           w       = workspace of length lw
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C           mpole = rescaled by 1/h_n(wavek0* radius) 
C           ier = error flag 
C                 0 normal return
C                 4 insufficient memory in w.
C
C---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,ier
      integer l,m,jj,kk
      integer lwfhs
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 ephi,imag,emul,sum,zmul
      complex *16 fhs(0:nterms),fhder(0:nterms)
      complex *16 wavek0,z
      data imag/(0.0d0,1.0d0)/
      z = wavek0*radius
      ifder = 0
      call h3dall(nterms,z,scale,fhs,ifder,fhder)
      do l=0,nterms
         do m=-l,l
	    zmul = fhs(l)
	    mpole(l,m) = mpole(l,m)/zmul
         enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3drescalestab(nterms,lmp,local,localn,
     1           radius,wavek0,scale,fjs,fjder,nbessel,ier)
C***********************************************************************
C
C     This subroutine takes as input the potential and its normal
C     derivative on a sphere of radius RADIUS and returns the 
C     j-expansion coefficients consist with the data (in a least
C     squares sense). That is 
C
C           phi  = sum local(n,m) j_n Y_n^m  ->
C           phin = sum local(n,m) wavek0 *j_n' Y_n^m and
C
C           local(n,m) * j_n        = phi_n^m
C           local(n,m) * j_n' *wavek0  = phin_n^m
C
C     The 1x1 normal equations are:
C
C           local(n,m)* ( j_n**2 + (wavek0* j_n')**2) = 
C                 
C                          j_n * phi_n^m + wavek0 * j_n' * phin_n^m.
C                   
C---------------------------------------------------------------------
C     INPUT:
C
C           nterms = order of spherical harmonic expansion
C           local = coefficients of s.h. expansion of phi
C           localn = coefficients of s.h. expansion of dphi/dn
C           radius = sphere radius
C           wavek0 = Helmholtz parameter
C           scale = scale parameter for expansions
C           w       = workspace of length lw
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C           local = computed as above.
C           ier = error flag 
C                 0 normal return
C                 4 insufficient memory in w.
C                 8 nbessel insufficient in calling jfuns3d.
C
C---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,ier
      integer l,m,jj,kk
      integer lwfhs
      complex *16 fjs(0:nbessel)
      complex *16 fjder(0:nterms)
      complex *16 local(0:lmp,-lmp:lmp)
      complex *16 localn(0:lmp,-lmp:lmp)
      complex *16 ephi,imag,emul,sum,zmul
      complex *16 wavek0,z,zh,zhn
      data imag/(0.0d0,1.0d0)/
      z = wavek0*radius
      ifder = 1
      call jfuns3d(ier1,nterms,z,scale,fjs,
     1             ifder,fjder,nbessel)
      if (ier1.eq.8) then
         ier = 8
	 return
      endif
      do l=0,nterms
         do m=-l,l
	    zh = fjs(l)
	    zhn = fjder(l)*wavek0
	    zmul = zh*zh + zhn*zhn
	    local(l,m) = (zh*local(l,m) + zhn*localn(l,m))/zmul
         enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dlocevalspherestab(local,wavek,scale,zshift,radius,
     1           nterms,nterms2,lmp,ynm,ynmd,phitemp,phitempn,
     2           nquad,xnodes,fjs,fjder,nbessel,ier)
C***********************************************************************
C
C     This subroutine evaluates a local expansion on a target
C     sphere at a distance (0,0,zshift) from the origin of radius 
C     "radius".
C
C---------------------------------------------------------------------
C     INPUT:
C
C     local    : coefficients of original multipole exp.
C     wavek       : Helmholtz parameter
C     scale    : scaling parameter
C     zshift   : shift distance along z-axis.
C     radius   : radius of sphere about (0,0,zshift)
C                              where phival is computed.
C     nterms   : number of terms in the orig. expansion
C     nterms2  : number of terms in exp on target sphere
C     lmp      : dimension param for local expansion
C     ynm      : storage for Ynm out to nterms.
C     nquad    : number of quadrature nodes
C                              on target sphere is nquad*nquad.
C     xnodes   : Legendre nodes x_j = cos theta_j.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     phitemp(i,j)  : jth mode of phi at ith quad node.
C     phitempn(i,j) : jth mode of phi at ith quad node.
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer nterms
      integer l,m,jnew,knew
      real *8 zshift, targ(3), center(3)
      real *8 xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 local(0:lmp,-lmp:lmp)
      complex *16 phitemp(nquad,-nterms2:nterms2)
      complex *16 phitempn(nquad,-nterms2:nterms2)
      complex *16 imag,pot,fld(3), wavek,z,uval,unval,ur,utheta
      complex *16 ephi1,ephik,ephi,fjs(0:nbessel),fjder(0:nbessel),ztmp1
      complex *16 ut1,ut2,ut3
      data imag/(0.0d0,1.0d0)/
C----- shift along z-axis.
C      note that everything is scaled.
      ier = 0
      pi = 4.0d0*datan(1.0d0)
      center(1) = 0.0d0
      center(2) = 0.0d0
      center(3) = 0.0d0
      iffld = 1
      ifder = 1
      do jj=1,nquad
      do m=-nterms2,nterms2
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (zshift+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (zshift+radius*ctheta)/rj
	 sthetaj = dsqrt(1.0d0-cthetaj**2)
	 rn = sthetaj*stheta + cthetaj*ctheta
	 thetan = (cthetaj*stheta - sthetaj*ctheta)/rj
	 z = wavek*rj
	 call ylgndr2s(nterms,cthetaj,ynm,ynmd)
	 call jfuns3d(jer,nterms,z,scale,fjs,ifder,fjder,nbessel)
         if (jer.eq.8) then
            ier = 8
	    return
         endif
	 do n = 0,nterms
	    fjder(n) = fjder(n)*wavek
         enddo
	 do n = 1,nterms
	    do m = 1,n
	       ynm(n,m) = ynm(n,m)*sthetaj
            enddo
         enddo
	 phitemp(jj,0) = local(0,0)*fjs(0)
	 phitempn(jj,0) = local(0,0)*fjder(0)*rn
         do n=1,nterms
	    phitemp(jj,0) = phitemp(jj,0) +
     1                local(n,0)*fjs(n)*ynm(n,0)
	    ut1 = fjder(n)*rn
	    ut2 = fjs(n)*thetan
	    ut3 = ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(jj,0) = phitempn(jj,0)+ut3*local(n,0)
	    do m=1,min(n,nterms2)
	       ztmp1 = fjs(n)*ynm(n,m)
	       phitemp(jj,m) = phitemp(jj,m) +
     1                local(n,m)*ztmp1
	       phitemp(jj,-m) = phitemp(jj,-m) +
     1                local(n,-m)*ztmp1
	       ut3 = ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(jj,m) = phitempn(jj,m)+ut3*local(n,m)
	       phitempn(jj,-m) = phitempn(jj,-m)+ut3*local(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dlocevalspherestab_fast(local,wavek,scale,zshift,
     1     radius,nterms,nterms2,lmp,ynm,ynmd,phitemp,phitempn,
     1     nquad,xnodes,fjs,fjder,nbessel,ier)
C***********************************************************************
C
C     This subroutine evaluates a local expansion on a target
C     sphere at a distance (0,0,zshift) from the origin of radius 
C     "radius".
C
C---------------------------------------------------------------------
C     INPUT:
C
C     local    : coefficients of original multipole exp.
C     wavek       : Helmholtz parameter
C     scale    : scaling parameter
C     zshift   : shift distance along z-axis.
C     radius   : radius of sphere about (0,0,zshift)
C                              where phival is computed.
C     nterms   : number of terms in the orig. expansion
C     nterms2  : number of terms in exp on target sphere
C     lmp      : dimension param for local expansion
C     ynm      : storage for Ynm out to nterms.
C     nquad    : number of quadrature nodes
C                              on target sphere is nquad*nquad.
C     xnodes   : Legendre nodes x_j = cos theta_j.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     phitemp(i,j)  : jth mode of phi at ith quad node.
C     phitempn(i,j) : jth mode of phi at ith quad node.
C
C***********************************************************************
      implicit real *8 (a-h,o-z)
      integer nterms
      integer l,m,jnew,knew
      real *8 zshift, targ(3), center(3)
      real *8 xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 local(0:lmp,-lmp:lmp)
      complex *16 phitemp(nquad,-nterms2:nterms2)
      complex *16 phitempn(nquad,-nterms2:nterms2)
      complex *16 imag,pot,fld(3), wavek,z,uval,unval,ur,utheta
      complex *16 ephi1,ephik,ephi,fjs(0:nbessel),fjder(0:nbessel),ztmp1
      complex *16 ut1,ut2,ut3
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
C----- shift along z-axis.
C      note that everything is scaled.
      ier = 0
      pi = 4.0d0*datan(1.0d0)
      center(1) = 0.0d0
      center(2) = 0.0d0
      center(3) = 0.0d0
      iffld = 1
      ifder = 1
      do jj=1,nquad
      do m=-nterms2,nterms2
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (zshift+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (zshift+radius*ctheta)/rj
	 sthetaj = dsqrt(1.0d0-cthetaj**2)
	 rn = sthetaj*stheta + cthetaj*ctheta
	 thetan = (cthetaj*stheta - sthetaj*ctheta)/rj
	 z = wavek*rj
	 call ylgndr2sf(nterms,cthetaj,ynm,ynmd,rat1,rat2)
	 call jfuns3d(jer,nterms,z,scale,fjs,ifder,fjder,nbessel)
         if (jer.eq.8) then
            ier = 8
	    return
         endif
	 do n = 0,nterms
	    fjder(n) = fjder(n)*wavek
         enddo
	 do n = 1,nterms
	    do m = 1,n
	       ynm(n,m) = ynm(n,m)*sthetaj
            enddo
         enddo
	 phitemp(jj,0) = local(0,0)*fjs(0)
	 phitempn(jj,0) = local(0,0)*fjder(0)*rn
         do n=1,nterms
	    phitemp(jj,0) = phitemp(jj,0) +
     1                local(n,0)*fjs(n)*ynm(n,0)
	    ut1 = fjder(n)*rn
	    ut2 = fjs(n)*thetan
	    ut3 = ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(jj,0) = phitempn(jj,0)+ut3*local(n,0)
	    do m=1,min(n,nterms2)
	       ztmp1 = fjs(n)*ynm(n,m)
	       phitemp(jj,m) = phitemp(jj,m) +
     1                local(n,m)*ztmp1
	       phitemp(jj,-m) = phitemp(jj,-m) +
     1                local(n,-m)*ztmp1
	       ut3 = ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(jj,m) = phitempn(jj,m)+ut3*local(n,m)
	       phitempn(jj,-m) = phitempn(jj,-m)+ut3*local(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dprojlocnmsep_fast
     1     (nterms,ldl,nquadn,ntold,xnodes,wts,phitemp,local)
C***********************************************************************
C
C     compute spherical harmonic expansion on unit sphere
C     of function tabulated at nquadn*nquadm grid points.
C
C---------------------------------------------------------------------
C     INPUT:
C
C           nterms = order of spherical harmonic expansion
C           ldl = dimension param for local expansion
C           nquadn = number of quadrature nodes in polar angle.
C           ntold =  number of azimuthal (m) modes in phitemp
C           xnodes = quad nodes in theta (nquadn of them)
C           wts = quad weights in theta (nquadn of them)
C           phitemp = tabulated function
C                    phitemp(i,j) = jth mode in phi at ith quad node 
C                                   in theta
C           ynm     = workspace for assoc Legendre functions
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C           local = coefficients of s.h. expansion
C
C    NOTE:
C
C    yrecursion.f produces Ynm with a nonstandard scaling:
C    (without the 1/sqrt(4*pi)). Thus the orthogonality relation
C    is
C             \int_S  Y_nm Y_n'm'*  dA = delta(n) delta(m) * 4*pi. 
C
C   This accounts for factor (1/2) = (2*pi) * 1/(4*pi) in zmul below.
C
C---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,nquadn,nquadm,ier
      integer l,m,jj,kk
      real *8 wts(1),xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      complex *16 wavek,phitemp(nquadn,-ntold:ntold)
      complex *16 local(0:ldl,-ldl:ldl)
      complex *16 ephi,imag,emul,sum,zmul,emul1
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      pi = 4.0d0*datan(1.0d0)
c     initialize local exp to zero
      do l = 0,ldl
         do m = -l,l
	    local(l,m) = 0.0d0
         enddo
      enddo
c     get local exp
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquadn
	 cthetaj = xnodes(jj)
	 call ylgndrf(nterms,cthetaj,ynm,rat1,rat2)
         do m=-ntold,ntold
	    zmul = phitemp(jj,m)*wts(jj)/2
            do l=abs(m),nterms
               local(l,m) = local(l,m) + 
     1   	       zmul*ynm(l,abs(m))
            enddo
         enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dmpevalspherenm_fast(mpole,wavek,scale,zshift,
     1     radius,nterms,nterms2,phitemp,nquad,xnodes,fhs,fhder)
C***********************************************************************
C
C     This subroutine evaluates a multipole expansion on a target
C     sphere at a distance (0,0,zshift) from the origin of radius 
C     "radius".
C
C---------------------------------------------------------------------
C     INPUT:
C
C     mpole    : coefficients of original multipole exp.
C     wavek       : Helmholtz parameter
C     scale    : mpole scaling parameter
C     zshift   : shift distance along z-axis.
C     radius   : radius of sphere about (0,0,zshift)
C                              where phival is computed.
C     nterms   : number of terms in the orig. expansion
C     ynm      : storage for assoc Legendre functions
C     phitemp  : storage for temporary array in O(p^3) scheme 
C     nquad    : number of quadrature nodes in theta
C     xnodes   : Legendre nodes x_j = cos theta_j.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     phival   : value of potential on tensor product
C                              mesh on target sphere.
C
C---------------------------------------------------------------------
      implicit none
      integer nterms,nterms2
      integer jj,l,m,n,jnew,knew,mabs,nquad
      real *8 radius,scale,zshift,ctheta,stheta,targ(3),center(3)
      real *8 cthetaj,rj,xnodes(nquad)
      real *8 ynm(0:nterms,0:nterms)
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 phitemp(nquad,-nterms:nterms)
      complex *16 imag,pot,fld(3),wavek,z
      complex *16 fhs(0:nterms),fhder(0:nterms)
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      center(1) = 0.0d0
      center(2) = 0.0d0
      center(3) = 0.0d0
      do jj=1,nquad
      do m=-nterms,nterms
         phitemp(jj,m) = 0.0d0
      enddo
      enddo
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (zshift+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (zshift+radius*ctheta)/rj
	 z = wavek*rj
	 call ylgndrf(nterms,cthetaj,ynm,rat1,rat2)
	 call h3dall(nterms,z,scale,fhs,0,fhder)
         do m=-nterms,nterms
	    mabs = abs(m)
	    do n=mabs,nterms
	       phitemp(jj,m) = phitemp(jj,m) +
     1                mpole(n,m)*fhs(n)*ynm(n,mabs)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dmpevalspherenmstab_fast(mpole,wavek,scale,zshift,
     1           radius,nterms,lmp,ynm,ynmd,phitemp,phitempn,
     2           nquad,xnodes,fhs,fhder)
C***********************************************************************
C
C     This subroutine evaluates a multipole expansion on a target
C     sphere at a distance (0,0,zshift) from the origin of radius 
C     "radius".
C
C---------------------------------------------------------------------
C     INPUT:
C
C     mpole    : coefficients of original multipole exp.
C     wavek       : Helmholtz parameter
C     scale    : mpole scaling parameter
C     zshift   : shift distance along z-axis.
C     radius   : radius of sphere about (0,0,zshift)
C                              where phival is computed.
C     nterms   : number of terms in the orig. expansion
C     lmp      : dimension param for mpole
C     ynm      : storage for assoc Legendre functions
C     ynmd     : storage for derivs of assoc Legendre functions
C     nquad    : number of quadrature nodes in theta
C     xnodes   : Legendre nodes x_j = cos theta_j.
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C     phitemp  : nquad by (-nterms,nterms)
C     phitempn : nquad by (-nterms,nterms)
C
C---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms
      integer l,m,jnew,knew
      real *8 zshift, targ(3), center(3)
      real *8 xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 mpole(0:lmp,-lmp:lmp)
      complex *16 phitemp(nquad,-nterms:nterms)
      complex *16 phitempn(nquad,-nterms:nterms)
      complex *16 imag,pot,fld(3), wavek,z,uval,unval,ur,utheta,ut1,ut2
      complex *16 ut3,ephi1,ephik,ephi,fhs(0:nterms),fhder(0:nterms)
      complex *16 ztmp1
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
C----- shift along z-axis.
C      note that everything is scaled.
      pi = 4.0d0*datan(1.0d0)
      center(1) = 0.0d0
      center(2) = 0.0d0
      center(3) = 0.0d0
      iffld = 1
      ifder = 1
      do jj=1,nquad
      do m=-nterms,nterms
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (zshift+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (zshift+radius*ctheta)/rj
	 sthetaj = dsqrt(1.0d0 - cthetaj**2)
	 rn = sthetaj*stheta + cthetaj*ctheta
	 thetan = (cthetaj*stheta - ctheta*sthetaj)/rj
	 z = wavek*rj
	 call ylgndr2sf(nterms,cthetaj,ynm,ynmd,rat1,rat2)
	 call h3dall(nterms,z,scale,fhs,ifder,fhder)
         do i = 0,nterms
	    fhder(i) = fhder(i)*wavek
         enddo
         do n=1,nterms
	    do m = 1,n
  	       ynm(n,m) = ynm(n,m)*sthetaj
              enddo
         enddo
	 phitemp(jj,0) = mpole(0,0)*fhs(0)
	 phitempn(jj,0) = mpole(0,0)*fhder(0)*rn
	 do n=1,nterms
	    phitemp(jj,0) = phitemp(jj,0) +
     1                mpole(n,0)*fhs(n)*ynm(n,0)
	    ut1 = fhder(n)*rn
	    ut2 = fhs(n)*thetan
	    ut3 = ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(jj,0) = phitempn(jj,0)+ut3*mpole(n,0)
            do m=1,n
	       ztmp1 = fhs(n)*ynm(n,m)
	       phitemp(jj,m) = phitemp(jj,m) +
     1                mpole(n,m)*ztmp1
	       phitemp(jj,-m) = phitemp(jj,-m) +
     1                mpole(n,-m)*ztmp1
	       ut3 = ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(jj,m) = phitempn(jj,m)+ut3*mpole(n,m)
	       phitempn(jj,-m) = phitempn(jj,-m)+ut3*mpole(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dprojlocsepstab_fast
     $          (nterms,ldl,nquadn,ntold,xnodes,wts,
     1           phitemp,phitempn,local,local2,ynm)
C***********************************************************************
C
C     compute spherical harmonic expansion on unit sphere
C     of function tabulated at nquadn*nquadm grid points.
C
C---------------------------------------------------------------------
C     INPUT:
C
C           nterms = order of spherical harmonic expansion
C           ldl = order of spherical harmonic expansion
C           nquadn = number of quadrature nodes in polar angle (theta).
C           ntold = number of modes in azimuthal direction.
C           xnodes = quad nodes in theta (polar angle) - nquadn of them
C           wts  = quad weights in theta (polar angle)
C           phitemp, phitempn = tabulated function and normal deriv
C                    phitemp(i,j) = jth mode of phi at ith quad node.
C                    phivaln(i,j) = jth mode of phi at ith quad node.
C
C           marray  = workspace 
C           ynm     = workspace for assoc Legendre functions
C
C---------------------------------------------------------------------
C     OUTPUT:
C
C           local  = coefficients of s.h. expansion for phi
C           local2 = coefficients of s.h. expansion for dphi/dn
C
C    NOTE:
C
C    yrecursion.f produces Ynm with a nonstandard scaling:
C    (without the 1/sqrt(4*pi)). Thus the orthogonality relation
C    is
C             \int_S  Y_nm Y_n'm'*  dA = delta(n) delta(m) * 4*pi. 
C
C   In the first loop below, you see
C
Cccc	    marray(jj,m) = sum*2*pi/nquad
C	    marray(jj,m) = sum/(2*nquad)
C
C   The latter has incorporated the 1/(4*pi) normalization factor
C   into the azimuthal quadrature weight (2*pi/nquad).
C
C---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,nquadn,nquadm,ier
      integer l,m,jj,kk
      real *8 wts(1),xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      complex *16 wavek
      complex *16 local(0:ldl,-ldl:ldl)
      complex *16 local2(0:ldl,-ldl:ldl)
      complex *16 phitemp(nquadn,-ntold:ntold)
      complex *16 phitempn(nquadn,-ntold:ntold)
      complex *16 ephi,imag,emul,sum,zmul,emul1
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      pi = 4.0d0*datan(1.0d0)
c     initialize local exp to zero
      do l = 0,ldl
         do m = -l,l
	    local(l,m) = 0.0d0
	    local2(l,m) = 0.0d0
         enddo
      enddo
c     get local exp
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquadn
	 cthetaj = xnodes(jj)
	 call ylgndrf(nterms,cthetaj,ynm,rat1,rat2)
         do m=-ntold,ntold
	    zmul = phitemp(jj,m)*wts(jj)/2.0d0
            do l=abs(m),nterms
               local(l,m) = local(l,m) + 
     1   	       zmul*ynm(l,abs(m))
            enddo
	    zmul = phitempn(jj,m)*wts(jj)/2.0d0
            do l=abs(m),nterms
               local2(l,m) = local2(l,m) + 
     1   	       zmul*ynm(l,abs(m))
            enddo
         enddo
      enddo
      return
      end
