      subroutine cart2polar(zat,r,theta,phi)
      implicit real *8 (a-h,o-z)
      real *8 zat(3)
      complex *16 ephi,eye
      data eye/(0.0d0,1.0d0)/
      r= sqrt(zat(1)**2+zat(2)**2+zat(3)**2)
      proj = sqrt(zat(1)**2+zat(2)**2)
      theta = datan2(proj,zat(3))
      if( abs(zat(1)) .eq. 0 .and. abs(zat(2)) .eq. 0 ) then
      phi = 0
      else
      phi = datan2(zat(2),zat(1))
      endif
      return
      end
c**********************************************************************
      subroutine P2P(ibox,target,pot,fld,jbox,source,charge,wavek)
c**********************************************************************
c     This subroutine calculates the potential POT and field FLD
c     at the target point TARGET, due to a charge at
c     SOURCE. The scaling is that required of the delta function
c     response: i.e.,
c
c              	pot = exp(i*k*r)/r
c		fld = -grad(pot)
c---------------------------------------------------------------------
c     INPUT:
c     source    : location of the source
c     charge    : charge strength
c     target    : location of the target
c     wavek        : helmholtz parameter
c---------------------------------------------------------------------
c     OUTPUT:
c     pot       : calculated potential
c     fld       : calculated gradient
c---------------------------------------------------------------------
      implicit none
      integer i,j,ibox(20),jbox(20)
      real *8 dx,dy,dz,R2,R,target(3,1000000),source(3,1000000)
      complex *16 i1,wavek,coef1,coef2
      complex *16 charge(1000000),pot(1000000),fld(3,1000000)
      data i1/(0.0d0,1.0d0)/
      do i=ibox(14),ibox(14)+ibox(15)-1
         do j=jbox(14),jbox(14)+jbox(15)-1
            dx=target(1,i)-source(1,j)
            dy=target(2,i)-source(2,j)
            dz=target(3,i)-source(3,j)
            R2=dx*dx+dy*dy+dz*dz
            if(R2.eq.0) cycle
            R=sqrt(R2)
            coef1=charge(j)*cdexp(i1*wavek*R)/R
            coef2=(1-i1*wavek*R)*coef1/R2
            pot(i)=pot(i)+coef1
            fld(1,i)=fld(1,i)+coef2*dx
            fld(2,i)=fld(2,i)+coef2*dy
            fld(3,i)=fld(3,i)+coef2*dz
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine P2M
     1     (ier,wavek,rscale,source,charge,ns,center,
     1     nterms,nterms1,lwfjs,mpole,wlege,nlege)
c***********************************************************************
c
c     Constructs multipole (h) expansion about CENTER due to NS sources
c     located at SOURCES(3,*).
c
c-----------------------------------------------------------------------
c     INPUT:
c
c     wavek              : Helmholtz parameter
c     scale           : the scaling factor.
c     sources         : coordinates of sources
c     charge          : source strengths
c     ns              : number of sources
c     center          : epxansion center
c     nterms          : order of multipole expansion
c     nterms1         : order of truncated expansion
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
c
c-----------------------------------------------------------------------
c     OUTPUT:
c
c     ier             : error return code
c     mpole           : coeffs of the h-expansion
c-----------------------------------------------------------------------
      implicit none
      integer i,j,m,l,n,ns,nterms,nterms1,nlege,ifder,lwfjs,jer,ntop,ier
      integer iscale(0:lwfjs)
      real *8 r,theta,phi,ctheta,stheta,cphi,sphi,wlege,rscale,dtmp
      real *8 thresh
      real *8 center(3),source(3,ns),dX(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 charge(ns),i1,wavek,z,ztmp,ephi1,ephi1inv
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjs(0:lwfjs),fjder(0:lwfjs)
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 mtemp(0:nterms,-nterms:nterms)
      data i1/(0.0d0,1.0d0)/
      data thresh/1.0d-15/
      ier=0
      do l = 0,nterms
         do m=-l,l
            mtemp(l,m) = 0
         enddo
      enddo
      do i = 1, ns
         dX(1)=source(1,i)-center(1)
         dX(2)=source(2,i)-center(2)
         dX(3)=source(3,i)-center(3)
         call cart2polar(dX,r,theta,phi)
         ctheta=dcos(theta)
         stheta=dsin(theta)
         cphi=dcos(phi)
         sphi=dsin(phi)
         ephi1=dcmplx(cphi,sphi)
         ephi(0)=1.0d0
         ephi(1)=ephi1
         ephi(-1)=dconjg(ephi1)
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi1
            ephi(-j)=ephi(-j+1)*ephi(-1)
         enddo
         call ylgndrfw(nterms1,ctheta,pp,wlege,nlege)
         ifder=0
         z=wavek*r
         call jfuns3d(jer,nterms1,z,rscale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
         do n = 0,nterms1
            fjs(n) = fjs(n)*charge(i)
         enddo
         mtemp(0,0)= mtemp(0,0) + fjs(0)
         do n=1,nterms1
            dtmp=pp(n,0)
            mtemp(n,0)= mtemp(n,0) + dtmp*fjs(n)
            do m=1,n
               ztmp=pp(n,m)*fjs(n)
               mtemp(n, m)= mtemp(n, m) + ztmp*dconjg(ephi(m))
               mtemp(n,-m)= mtemp(n,-m) + ztmp*dconjg(ephi(-m))
            enddo
         enddo
      enddo
      do l = 0,nterms
         do m=-l,l
            mpole(l,m) = mpole(l,m)+mtemp(l,m)*i1*wavek
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2M(wavek,scale,x0y0z0,mpole,nterms,scale2,
     1           xnynzn,mpolen,nterms2,ldc,
     2           radius,xnodes,wts,nquad,nq,ier)
c***********************************************************************
c     Shift multipole expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     x0y0z0  : center of original multiple expansion
c     xnynzn  : center of shifted expansion
c     mpole   : coefficients of original multiple expansion
c     nterms  : order of multipole expansion
c     nterms2 : order of shifted expansion
c     scale   : scaling parameter for mpole expansion
c     scale2  : scaling parameter for shifted expansion
c     radius  : radius of sphere on which shifted expansion is computed
c     xnodes  : Legendre nodes (precomputed)
c     wts     : Legendre weights (precomputed)
c     nquad   : number of quadrature nodes in theta
c     nq      : used to allocate work arrays for both z-shift and rotations.
c---------------------------------------------------------------------
c     OUTPUT:
c     mpolen  : coefficients of shifted expansion
c---------------------------------------------------------------------
      implicit none
      integer  nterms, lw, lused, ier, nq, nquad, nquse,ldc,nterms2
      real *8 x0y0z0(3),xnynzn(3)
      real *8 radius, zshift
      real *8 xnodes(1),wts(1)
      real *8 d,theta,ctheta,phi,scale,scale2,rvec(3)
      real *8 ynm(0:ldc,0:ldc)
      real *8 ynmd(0:ldc,0:ldc)
      complex *16 phitemp(nq,-ldc:ldc)
      complex *16 phitemp2(nq,-ldc:ldc)
      complex *16 fhs(0:nterms)
      complex *16 fhder(0:nterms)
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 marray1(0:nterms,-nterms:nterms)
      complex *16 mpolen(0:nterms2,-nterms2:nterms2)
      complex *16 marray(0:ldc,-ldc:ldc)
      complex *16 wavek
      complex *16 ephi(-ldc-1:ldc+1),imag
      complex *16, allocatable :: mptemp(:,:)
      integer  l,m,jnew,knew
      data imag/(0.0d0,1.0d0)/
      allocate( mptemp(0:nterms2,-nterms2:nterms2) )
      rvec(1) = xnynzn(1) - x0y0z0(1)
      rvec(2) = xnynzn(2) - x0y0z0(2)
      rvec(3) = xnynzn(3) - x0y0z0(3)
      call cart2polar(rvec,d,theta,phi)
      ephi(1) = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do l = 1,ldc
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,nterms
         do m=-l,l
            marray1(l,m)=mpole(l,m)*ephi(m)
         enddo
      enddo
      do l=0,nterms2
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( nterms .ge. 30 ) then
      call rotviaprojf90(theta,nterms,nterms,nterms,marray1,nterms,
     1        marray,ldc)
      else
      call rotviarecur3f90(theta,nterms,nterms,nterms,marray1,nterms,
     1        marray,ldc)
      endif
      zshift = d
      call h3dmpevalspherenm_fast(marray,wavek,scale,
     1     zshift,radius,nterms,ldc,ynm,
     1     phitemp,nquad,xnodes,fhs,fhder)
      call h3dprojlocnmsep_fast
     1     (nterms2,nterms2,nquad,nterms,xnodes,wts,
     1     phitemp,mptemp,ynm)
      call h3drescalemp(nterms2,nterms2,mptemp,radius,wavek,
     1     scale2,fhs,fhder)
      if( nterms2 .ge. 30 ) then
      call rotviaprojf90(-theta,nterms2,nterms2,nterms2,mptemp,
     1        nterms2,marray,ldc)
      else
      call rotviarecur3f90(-theta,nterms2,nterms2,nterms2,mptemp,
     1        nterms2,marray,ldc)
      endif
      do l=0,nterms2
         do m=-l,l
            mptemp(l,m)=ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,min(ldc,nterms2)
         do m=-l,l
            mpolen(l,m) = mpolen(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2L(wavek,scale,x0y0z0,mpole,
     1     nterms,scale2,xnynzn,local,nterms2,nterms_trunc,
     1     radius,xnodes,wts,nquad,nq,lwfjs,ier)
c***********************************************************************
c     Convert multipole expansion to a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek  : Helmholtz parameter
c     x0y0z0 : center of original multiple expansion
c     xnynzn : center of shifted local expansion
c     mpole  : coefficients of original multiple expansion
c     nterms : order of multipole expansion
c     scale  : scaling parameter for mpole expansion
c     scale2 : scaling parameter for local expansion
c     radius : radius of sphere on which local expansion is computed
c     xnodes : Legendre nodes (precomputed)
c     wts    : Legendre weights (precomputed)
c     nquad  : number of quadrature nodes used (really nquad**2)
c---------------------------------------------------------------------
c     OUTPUT:
c     local : coefficients of shifted local expansion
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer  nterms,ier,l,m,jnew,knew
      integer  iscale(0:lwfjs)
      real *8 d,theta,ctheta,phi,scale,scale2
      real *8 x0y0z0(3),xnynzn(3)
      real *8 xnodes(1),wts(1),rvec(3)
      real *8 zshift
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 phitemp(nq,-nterms:nterms)
      complex *16 phitempn(nq,-nterms:nterms)
      complex *16 mp2(0:nterms,-nterms:nterms)
      complex *16 fhs(0:nterms)
      complex *16 fhder(0:nterms)
      complex *16 fjs(0:lwfjs)
      complex *16 fjder(0:lwfjs)
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 marray1(0:nterms_trunc,-nterms_trunc:nterms_trunc)
      complex *16 local(0:nterms2,-nterms2:nterms2)
      complex *16 mptemp(0:nterms_trunc,-nterms_trunc:nterms_trunc)
      complex *16 marray(0:nterms,-nterms:nterms)
      complex *16 wavek
      complex *16 ephi(-nterms-1:nterms+1),imag      
      data imag/(0.0d0,1.0d0)/
      rvec(1) = xnynzn(1) - x0y0z0(1)
      rvec(2) = xnynzn(2) - x0y0z0(2)
      rvec(3) = xnynzn(3) - x0y0z0(3)
      call cart2polar(rvec,d,theta,phi)
      ephi(1) = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do l = 1,nterms
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,nterms_trunc
         do mp=-l,l
            marray1(l,mp) = mpole(l,mp)*ephi(mp)
         enddo
      enddo
      do l=0,nterms_trunc
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( nterms_trunc .ge. 30 ) then
      call rotviaprojf90(theta,nterms_trunc,nterms_trunc,nterms_trunc,
     1     marray1,nterms_trunc,marray,nterms)
      else
      call rotviarecur3f90(theta,nterms_trunc,nterms_trunc,nterms_trunc,
     1     marray1,nterms_trunc,marray,nterms)
      endif
      zshift = d
      call h3dmpevalspherenmstab_fast(marray,wavek,scale,zshift,radius,
     2     nterms_trunc,nterms,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     3     fhs,fhder)
      call h3dprojlocsepstab_fast
     $   (nterms_trunc,nterms_trunc,nquad,nterms_trunc,xnodes,wts,
     1     phitemp,phitempn,mptemp,mp2,ynm)
      call h3drescalestab(nterms_trunc,nterms_trunc,mptemp,mp2,
     1     radius,wavek,scale2,fjs,fjder,iscale,lwfjs,ier)
      if( nterms_trunc .ge. 30 ) then
         call rotviaprojf90(-theta,nterms_trunc,nterms_trunc,
     1        nterms_trunc,mptemp,nterms_trunc,marray,nterms)
      else
         call rotviarecur3f90(-theta,nterms_trunc,nterms_trunc,
     1        nterms_trunc,mptemp,nterms_trunc,marray,nterms)
      endif
      do l=0,nterms_trunc
         do m=-l,l
            mptemp(l,m) = ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,nterms_trunc
         do m=-l,l
            local(l,m) = local(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
      subroutine L2L(wavek,scale,x0y0z0,locold,nterms,
     1           scale2,xnynzn,local,nterms2,ldc,
     2           radius,xnodes,wts,nquad,nq,lwfjs,ier)
c***********************************************************************
c     Shifts center of a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     scale   : scaling parameter for locold expansion
c     x0y0z0  : center of original multiple expansion
c     locold  : coefficients of original multiple expansion
c     nterms  : order of original local expansion
c     scale2  : scaling parameter for local expansion
c     xnynzn  : center of shifted local expansion
c     nterms2 : order of new local expansion
c     marray  : work array
c     dc      : another work array
c     ldc     : dimension parameter for marray and ldc
c     rd1     : work array for rotation operators.
c     rd2     : work array for rotation operators.
c     ephi    : work array for rotation operators.
c     radius  : radius of sphere on which local expansion is computed
c     xnodes  : Legendre nodes (precomputed)
c     wts     : Legendre weights (precomputed)
c     nquad   : number of quadrature nodes in theta direction.
c---------------------------------------------------------------------
c     OUTPUT:
c     local   : coefficients of shifted local expansion
c***********************************************************************
      implicit real *8 (a-h,o-z)
      integer nterms,ier,l,m,jnew,knew
      integer iscale(0:lwfjs)
      real *8 x0y0z0(3),xnynzn(3),rvec(3)
      real *8 xnodes(1),wts(1)
      real *8 d,theta,ctheta,phi,scale,scale2
      real *8 ynm(0:ldc,0:ldc)
      real *8 ynmd(0:ldc,0:ldc)
      complex *16 phitemp(nq,-ldc:ldc)
      complex *16 phitempn(nq,-ldc:ldc)
      complex *16 mp2(0:ldc,-ldc:ldc)
      complex *16 fjs(0:lwfjs)
      complex *16 fjder(0:lwfjs)
      complex *16 locold(0:nterms,-nterms:nterms)
      complex *16 local(0:nterms2,-nterms2:nterms2)
      complex *16 mptemp(0:nterms2,-nterms2:nterms2)
      complex *16 marray(0:ldc,-ldc:ldc)
      complex *16 marray1(0:nterms,-nterms:nterms)
      complex *16 wavek,imag,ephi1
      complex *16 ephi(-ldc-1:ldc+1)
      data imag/(0.0d0,1.0d0)/
      rvec(1) = xnynzn(1) - x0y0z0(1)
      rvec(2) = xnynzn(2) - x0y0z0(2)
      rvec(3) = xnynzn(3) - x0y0z0(3)
      call cart2polar(rvec,d,theta,phi)
      ephi1 = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=dconjg(ephi1)
      do l = 1,ldc
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,nterms
         do mp=-l,l
            marray1(l,mp) = locold(l,mp)*ephi(mp)
         enddo
      enddo
      do l=0,nterms2
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( nterms2 .ge. 30 ) then
      call rotviaprojf90(theta,nterms,nterms,nterms2,marray1,nterms,
     1      marray,ldc)
      else
      call rotviarecur3f90(theta,nterms,nterms,nterms2,marray1,
     1      nterms,marray,ldc)
      endif
      zshift = d
      call h3dlocevalspherestab_fast(marray,wavek,scale,
     1     zshift,radius,nterms,nterms2,
     1     ldc,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     1     iscale,fjs,fjder,lwfjs,ier)
      call h3dprojlocsepstab_fast
     1     (nterms2,nterms2,nquad,nterms2,xnodes,wts,
     1     phitemp,phitempn,mptemp,mp2,ynm)
      call h3drescalestab(nterms2,nterms2,mptemp,mp2,
     1      radius,wavek,scale2,fjs,fjder,iscale,lwfjs,ier)
      if( nterms2 .ge. 30 ) then
      call rotviaprojf90(-theta,nterms2,nterms2,nterms2,mptemp,
     1      nterms2,marray,ldc)
      else
      call rotviarecur3f90(-theta,nterms2,nterms2,nterms2,mptemp,
     1      nterms2,marray,ldc)
      endif
      do l=0,nterms2
         do m=-l,l
            mptemp(l,m)=ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,min(ldc,nterms2)
         do m=-l,l
            local(l,m) = local(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
c**********************************************************************
      subroutine L2P(wavek,rscale,center,locexp,nterms,
     1     nterms1,lwfjs,target,nt,pot,fld,wlege,nlege,ier)
c**********************************************************************
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point TARGET.
c
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c---------------------------------------------------------------------
c     INPUT:
c     wavek      : the Helmholtz coefficient
c     rscale     : scaling parameter used in forming expansion
c     center     : coordinates of the expansion center
c     locexp     : coeffs of the j-expansion
c     nterms     : order of the h-expansion
c     nterms1    : order of the truncated expansion
c     target(3)   : target vector
c     nt         : number of targets
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
c---------------------------------------------------------------------
c     OUTPUT:
c     ier        : error return code
c     pot        : potential at target (if requested)
c     fld(3)     : gradient at target (if requested)
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c---------------------------------------------------------------------
      implicit none
      integer i,j,m,n,nt,ier,jer,nterms,nterms1,nlege,ntop,lwfjs
      integer iscale(0:lwfjs)
      real *8 r,rx,ry,rz,theta,thetax,thetay,thetaz,rscale
      real *8 phi,phix,phiy,phiz,ctheta,stheta,cphi,sphi,wlege
      real *8 center(3),target(3,1),dX(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 wavek,pot(1),fld(3,1),ephi1,ephi1inv
      complex *16 locexp(0:nterms,-nterms:nterms)
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjsuse,fjs(0:lwfjs),fjder(0:lwfjs)
      complex *16 eye,ur,utheta,uphi,ztmp,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
      data eye/(0.0d0,1.0d0)/
      ier=0
      do i=1,nt
         dX(1)=target(1,i)-center(1)
         dX(2)=target(2,i)-center(2)
         dX(3)=target(3,i)-center(3)
         call cart2polar(dX,r,theta,phi)
         ctheta = dcos(theta)
         stheta=sqrt(1-ctheta*ctheta)
         cphi = dcos(phi)
         sphi = dsin(phi)
         ephi1 = dcmplx(cphi,sphi)
         ephi(0)=1.0d0
         ephi(1)=ephi1
         ephi(-1)=dconjg(ephi1)
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi1
            ephi(-j)=ephi(-j+1)*ephi(-1)
         enddo
         rx = stheta*cphi
         thetax = ctheta*cphi
         phix = -sphi
         ry = stheta*sphi
         thetay = ctheta*sphi
         phiy = cphi
         rz = ctheta
         thetaz = -stheta
         phiz = 0.0d0
         call ylgndr2sfw(nterms1,ctheta,pp,ppd,wlege,nlege)
         z=wavek*r
         call jfuns3d(jer,nterms1,z,rscale,fjs,1,fjder,
     1	      lwfjs,iscale,ntop)
         if (jer.ne.0) then
            ier=8
            return
         endif
         pot(i)=pot(i)+locexp(0,0)*fjs(0)
         do j=0,nterms1
            fjder(j)=fjder(j)*wavek
         enddo
         ur = locexp(0,0)*fjder(0)
         utheta = 0.0d0
         uphi = 0.0d0
         do n=1,nterms1
            pot(i)=pot(i)+locexp(n,0)*fjs(n)*pp(n,0)
            ur = ur + fjder(n)*pp(n,0)*locexp(n,0)
            fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
            fjsuse = wavek*fjsuse/(2*n+1.0d0)
            utheta = utheta -locexp(n,0)*fjsuse*ppd(n,0)*stheta
            do m=1,n
               ztmp1=fjs(n)*pp(n,m)*stheta
               ztmp2 = locexp(n,m)*ephi(m)
               ztmp3 = locexp(n,-m)*ephi(-m)
               ztmpsum = ztmp2+ztmp3
               pot(i)=pot(i)+ztmp1*ztmpsum
               ur = ur + fjder(n)*pp(n,m)*stheta*ztmpsum
               utheta = utheta -ztmpsum*fjsuse*ppd(n,m)
               ztmpsum = eye*m*(ztmp2 - ztmp3)
               uphi = uphi + fjsuse*pp(n,m)*ztmpsum
            enddo
         enddo
         ux = ur*rx + utheta*thetax + uphi*phix
         uy = ur*ry + utheta*thetay + uphi*phiy
         uz = ur*rz + utheta*thetaz + uphi*phiz
         fld(1,i) = fld(1,i)-ux
         fld(2,i) = fld(2,i)-uy
         fld(3,i) = fld(3,i)-uz
      enddo
      return
      end

