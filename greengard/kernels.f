      subroutine cart2sph(dX,r,theta,phi)
      implicit none
      real *8 dX(3),r,theta,phi
      r = sqrt(dX(1)*dX(1)+dX(2)*dX(2)+dX(3)*dX(3))
      theta = datan2(sqrt(dX(1)*dX(1)+dX(2)*dX(2)),dX(3))
      if(abs(dX(1)).eq.0.and.abs(dX(2)).eq.0) then
         phi = 0
      else
         phi = datan2(dX(2),dX(1))
      endif
      return
      end
c**********************************************************************
      subroutine P2P(ibox,Xi,pi,Fi,jbox,Xj,qj,wavek)
c**********************************************************************
c     This subroutine calculates the potential and field
c     at the target point Xi, due to a charge at Xj.
c     The scaling is that required of the delta function response: i.e.,
c              	pi = exp(i*k*r)/r
c		Fi = -grad(pi)
c---------------------------------------------------------------------
c     INPUT:
c     Xj    : location of the source
c     qj    : charge strength
c     Xi    : location of the target
c     wavek : helmholtz parameter
c---------------------------------------------------------------------
c     OUTPUT:
c     pi   : calculated potential
c     Fi   : calculated gradient
c---------------------------------------------------------------------
      implicit none
      integer i,j,ibox(20),jbox(20)
      real *8 dX(3),R2,R,Xi(3,1000000),Xj(3,1000000)
      complex *16 imag,wavek,coef1,coef2
      complex *16 qj(1000000),pi(1000000),Fi(3,1000000)
      data imag/(0.0d0,1.0d0)/
      do i=ibox(14),ibox(14)+ibox(15)-1
         do j=jbox(14),jbox(14)+jbox(15)-1
            dX(1)=Xi(1,i)-Xj(1,j)
            dX(2)=Xi(2,i)-Xj(2,j)
            dX(3)=Xi(3,i)-Xj(3,j)
            R2=dX(1)*dX(1)+dX(2)*dX(2)+dX(3)*dX(3)
            if(R2.eq.0) cycle
            R=sqrt(R2)
            coef1=qj(j)*cdexp(imag*wavek*R)/R
            coef2=(1-imag*wavek*R)*coef1/R2
            pi(i)=pi(i)+coef1
            Fi(1,i)=Fi(1,i)+coef2*dX(1)
            Fi(2,i)=Fi(2,i)+coef2*dX(2)
            Fi(3,i)=Fi(3,i)+coef2*dX(3)
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine P2M(ier,wavek,scale,Xj,qj,nj,Xi,
     1     nterms,ntrunc,nbessel,Mi,Anm,Pmax)
c***********************************************************************
c     Constructs multipole expansion about Xi due to nj sources
c     located at Xj.
c-----------------------------------------------------------------------
c     INPUT:
c     wavek  : Helmholtz parameter
c     scale  : the scaling factor.
c     Xj     : coordinates of sources
c     qj     : source strengths
c     nj     : number of sources
c     Xi     : epxansion center
c     nterms : order of multipole expansion
c     ntrunc : order of truncated expansion
c     Anm    : precomputed array of scaling coeffs for Pnm
c     Pmax   : dimension parameter for Anm
c-----------------------------------------------------------------------
c     OUTPUT:
c     Mi     : coeffs of the h-expansion
c-----------------------------------------------------------------------
      implicit none
      integer i,m,n,nj,nterms,ntrunc,Pmax,nbessel,ier
      real *8 r,theta,phi,ctheta,stheta,Anm,scale
      real *8 Xi(3),Xj(3,nj),dX(3)
      real *8 Ynm(0:nterms,0:nterms)
      complex *16 qj(nj),imag,wavek,z,Ynmjn
      complex *16 ephi(ntrunc)
      complex *16 jn(0:nbessel),jnd(0:nbessel)
      complex *16 Mi(0:nterms,-nterms:nterms)
      complex *16 Mnm(0:nterms,-nterms:nterms)
      data imag/(0.0d0,1.0d0)/
      ier=0
      do n=0,nterms
         do m=-n,n
            Mnm(n,m) = 0
         enddo
      enddo
      do i=1,nj
         dX(1)=Xj(1,i)-Xi(1)
         dX(2)=Xj(2,i)-Xi(2)
         dX(3)=Xj(3,i)-Xi(3)
         call cart2sph(dX,r,theta,phi)
         ctheta=dcos(theta)
         stheta=dsin(theta)
         ephi(1)=exp(imag*phi)
         do n=2,ntrunc
            ephi(n)=ephi(n-1)*ephi(1)
         enddo
         call ylgndrfw(ntrunc,ctheta,Ynm,Anm,Pmax)
         z=wavek*r
         call jfuns3d(ier,ntrunc,z,scale,jn,0,jnd,nbessel)
         do n = 0,ntrunc
            jn(n)=jn(n)*qj(i)
         enddo
         Mnm(0,0)=Mnm(0,0)+jn(0)
         do n=1,ntrunc
            Mnm(n,0)=Mnm(n,0)+Ynm(n,0)*jn(n)
            do m=1,n
               Ynmjn=Ynm(n,m)*jn(n)
               Mnm(n, m)=Mnm(n, m)+Ynmjn*dconjg(ephi(m))
               Mnm(n,-m)=Mnm(n,-m)+Ynmjn*ephi(m)
            enddo
         enddo
      enddo
      do n=0,nterms
         do m=-n,n
            Mi(n,m)=Mi(n,m)+Mnm(n,m)*imag*wavek
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2M(wavek,scalej,Xj,Mj,ntermsj,
     1     scalei,Xi,Mi,ntermsi,
     2     radius,xnodes,wts,nquad,ier)
c***********************************************************************
c     Shift multipole expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     Xj      : center of original multiple expansion
c     Xi      : center of shifted expansion
c     Mj      : coefficients of original multiple expansion
c     ntermsj : order of multipole expansion
c     ntermsi : order of shifted expansion
c     scalej  : scaling parameter for mpole expansion
c     scalei  : scaling parameter for shifted expansion
c     radius  : radius of sphere on which shifted expansion is computed
c     xnodes  : Legendre nodes (precomputed)
c     wts     : Legendre weights (precomputed)
c     nquad   : number of quadrature nodes in theta
c     nq      : used to allocate work arrays for both z-shift and rotations.
c---------------------------------------------------------------------
c     OUTPUT:
c     Mi  : coefficients of shifted expansion
c---------------------------------------------------------------------
      implicit none
      integer l,m,jnew,knew
      integer ntermsj,ier,nquad,ntermsi
      real *8 Xj(3),Xi(3)
      real *8 radius, zshift
      real *8 xnodes(nquad),wts(nquad)
      real *8 d,theta,ctheta,phi,scalej,scalei,dX(3)
      complex *16 Mj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 marray1(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mi(0:ntermsi,-ntermsi:ntermsi)
      complex *16 marray(0:ntermsj,-ntermsj:ntermsj)
      complex *16 phitemp(nquad,-ntermsj:ntermsj)
      complex *16 wavek
      complex *16 ephi(-ntermsj-1:ntermsj+1),imag
      complex *16, allocatable :: mptemp(:,:)
      data imag/(0.0d0,1.0d0)/
      allocate( mptemp(0:ntermsi,-ntermsi:ntermsi) )
      dX(1) = Xi(1) - Xj(1)
      dX(2) = Xi(2) - Xj(2)
      dX(3) = Xi(3) - Xj(3)
      call cart2sph(dX,d,theta,phi)
      ephi(1) = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do l = 1,ntermsj
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,ntermsj
         do m=-l,l
            marray1(l,m)=Mj(l,m)*ephi(m)
         enddo
      enddo
      do l=0,ntermsi
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( ntermsj .ge. 30 ) then
         call rotviaprojf90(theta,ntermsj,marray1,ntermsj,
     1        marray,ntermsj)
      else
         call rotviarecur3f90(theta,ntermsj,marray1,
     1        ntermsj,marray,ntermsj)
      endif
      zshift = d
      call h3dmpevalspherenm_fast(marray,wavek,scalej,
     1     zshift,radius,ntermsj,phitemp,
     1     nquad,xnodes)
      call h3dprojlocnmsep_fast
     1     (ntermsi,ntermsi,nquad,ntermsj,xnodes,wts,
     1     phitemp,mptemp)
      call h3drescalemp(ntermsi,ntermsi,mptemp,radius,wavek,
     1     scalei)
      if( ntermsi .ge. 30 ) then
         call rotviaprojf90(-theta,ntermsi,mptemp,
     1        ntermsi,marray,ntermsj)
      else
         call rotviarecur3f90(-theta,ntermsi,mptemp,
     1        ntermsi,marray,ntermsj)
      endif
      do l=0,ntermsi
         do m=-l,l
            mptemp(l,m)=ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,min(ntermsj,ntermsi)
         do m=-l,l
            Mi(l,m) = Mi(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2L(wavek,scalej,Xj,Mj,
     1     ntermsj,scalei,Xi,local,ntermsi,ntrunc,
     1     radius,xnodes,wts,nquad,nq,nbessel,ier)
c***********************************************************************
c     Convert multipole expansion to a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek  : Helmholtz parameter
c     Xj : center of original multiple expansion
c     Xi : center of shifted local expansion
c     Mj  : coefficients of original multiple expansion
c     ntermsj : order of multipole expansion
c     scalej  : scaling parameter for mpole expansion
c     scalei : scaling parameter for local expansion
c     radius : radius of sphere on which local expansion is computed
c     xnodes : Legendre nodes (precomputed)
c     wts    : Legendre weights (precomputed)
c     nquad  : number of quadrature nodes used (really nquad**2)
c---------------------------------------------------------------------
c     OUTPUT:
c     local : coefficients of shifted local expansion
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer ntermsj,ier,l,m,jnew,knew
      real *8 d,theta,ctheta,phi,scalej,scalei
      real *8 Xj(3),Xi(3)
      real *8 xnodes(1),wts(1),dX(3)
      real *8 zshift
      real *8 ynm(0:ntermsj,0:ntermsj)
      real *8 ynmd(0:ntermsj,0:ntermsj)
      complex *16 phitemp(nq,-ntermsj:ntermsj)
      complex *16 phitempn(nq,-ntermsj:ntermsj)
      complex *16 mp2(0:ntermsj,-ntermsj:ntermsj)
      complex *16 fhs(0:ntermsj)
      complex *16 fhder(0:ntermsj)
      complex *16 jn(0:nbessel)
      complex *16 jnd(0:nbessel)
      complex *16 Mj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 marray1(0:ntrunc,-ntrunc:ntrunc)
      complex *16 local(0:ntermsi,-ntermsi:ntermsi)
      complex *16 mptemp(0:ntrunc,-ntrunc:ntrunc)
      complex *16 marray(0:ntermsj,-ntermsj:ntermsj)
      complex *16 wavek
      complex *16 ephi(-ntermsj-1:ntermsj+1),imag      
      data imag/(0.0d0,1.0d0)/
      dX(1) = Xi(1) - Xj(1)
      dX(2) = Xi(2) - Xj(2)
      dX(3) = Xi(3) - Xj(3)
      call cart2sph(dX,d,theta,phi)
      ephi(1) = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do l = 1,ntermsj
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,ntrunc
         do mp=-l,l
            marray1(l,mp) = Mj(l,mp)*ephi(mp)
         enddo
      enddo
      do l=0,ntrunc
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( ntrunc .ge. 30 ) then
         call rotviaprojf90(theta,ntrunc,
     1        marray1,ntrunc,marray,ntermsj)
      else
         call rotviarecur3f90(theta,ntrunc,
     1        marray1,ntrunc,marray,ntermsj)
      endif
      zshift = d
      call h3dmpevalspherenmstab_fast(marray,wavek,scalej,zshift,radius,
     2     ntrunc,ntermsj,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     3     fhs,fhder)
      call h3dprojlocsepstab_fast
     $   (ntrunc,ntrunc,nquad,ntrunc,xnodes,wts,
     1     phitemp,phitempn,mptemp,mp2,ynm)
      call h3drescalestab(ntrunc,ntrunc,mptemp,mp2,
     1     radius,wavek,scalei,jn,jnd,nbessel,ier)
      if( ntrunc .ge. 30 ) then
         call rotviaprojf90(-theta,ntrunc,
     1        mptemp,ntrunc,marray,ntermsj)
      else
         call rotviarecur3f90(-theta,ntrunc,
     1        mptemp,ntrunc,marray,ntermsj)
      endif
      do l=0,ntrunc
         do m=-l,l
            mptemp(l,m) = ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,ntrunc
         do m=-l,l
            local(l,m) = local(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
      subroutine L2L(wavek,scalej,Xj,locold,ntermsj,
     1           scalei,Xi,local,ntermsi,ldc,
     2           radius,xnodes,wts,nquad,nq,nbessel,ier)
c***********************************************************************
c     Shifts center of a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     scalej   : scaling parameter for locold expansion
c     Xj  : center of original multiple expansion
c     locold  : coefficients of original multiple expansion
c     ntermsj  : order of original local expansion
c     scalei  : scaling parameter for local expansion
c     Xi  : center of shifted local expansion
c     ntermsi : order of new local expansion
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
      integer ntermsj,ier,l,m,jnew,knew
      real *8 Xj(3),Xi(3),dX(3)
      real *8 xnodes(1),wts(1)
      real *8 d,theta,ctheta,phi,scalej,scalei
      real *8 ynm(0:ldc,0:ldc)
      real *8 ynmd(0:ldc,0:ldc)
      complex *16 phitemp(nq,-ldc:ldc)
      complex *16 phitempn(nq,-ldc:ldc)
      complex *16 mp2(0:ldc,-ldc:ldc)
      complex *16 jn(0:nbessel)
      complex *16 jnd(0:nbessel)
      complex *16 locold(0:ntermsj,-ntermsj:ntermsj)
      complex *16 local(0:ntermsi,-ntermsi:ntermsi)
      complex *16 mptemp(0:ntermsi,-ntermsi:ntermsi)
      complex *16 marray(0:ldc,-ldc:ldc)
      complex *16 marray1(0:ntermsj,-ntermsj:ntermsj)
      complex *16 wavek,imag
      complex *16 ephi(-ldc-1:ldc+1)
      data imag/(0.0d0,1.0d0)/
      dX(1) = Xi(1) - Xj(1)
      dX(2) = Xi(2) - Xj(2)
      dX(3) = Xi(3) - Xj(3)
      call cart2sph(dX,d,theta,phi)
      ephi(0)=1.0d0
      ephi(1)=exp(imag*phi)
      ephi(-1)=dconjg(ephi(1))
      do l = 1,ldc
         ephi(l+1) = ephi(l)*ephi(1)
         ephi(-1-l) = dconjg(ephi(l+1))
      enddo
      do l=0,ntermsj
         do mp=-l,l
            marray1(l,mp) = locold(l,mp)*ephi(mp)
         enddo
      enddo
      do l=0,ntermsi
         do m=-l,l
            mptemp(l,m)=0.0d0
         enddo
      enddo
      if( ntermsi .ge. 30 ) then
         call rotviaprojf90(theta,ntermsj,marray1,ntermsj,
     1        marray,ldc)
      else
         call rotviarecur3f90(theta,ntermsj,marray1,
     1        ntermsj,marray,ldc)
      endif
      zshift = d
      call h3dlocevalspherestab_fast(marray,wavek,scalej,
     1     zshift,radius,ntermsj,ntermsi,
     1     ldc,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     1     jn,jnd,nbessel,ier)
      call h3dprojlocsepstab_fast
     1     (ntermsi,ntermsi,nquad,ntermsi,xnodes,wts,
     1     phitemp,phitempn,mptemp,mp2,ynm)
      call h3drescalestab(ntermsi,ntermsi,mptemp,mp2,
     1      radius,wavek,scalei,jn,jnd,nbessel,ier)
      if( ntermsi .ge. 30 ) then
         call rotviaprojf90(-theta,ntermsi,mptemp,
     1        ntermsi,marray,ldc)
      else
         call rotviarecur3f90(-theta,ntermsi,mptemp,
     1        ntermsi,marray,ldc)
      endif
      do l=0,ntermsi
         do m=-l,l
            mptemp(l,m)=ephi(-m)*marray(l,m)
         enddo
      enddo
      do l = 0,min(ldc,ntermsi)
         do m=-l,l
            local(l,m) = local(l,m)+mptemp(l,m)
         enddo
      enddo
      return
      end
c**********************************************************************
      subroutine L2P(wavek,scalej,center,locexp,nterms,
     1     ntrunc,nbessel,Xi,nt,pot,fld,Anm,Pmax,ier)
c**********************************************************************
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point TARGET.
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c---------------------------------------------------------------------
c     INPUT:
c     wavek  : the Helmholtz coefficient
c     scalej  : scaling parameter used in forming expansion
c     center : coordinates of the expansion center
c     locexp : coeffs of the j-expansion
c     nterms : order of the h-expansion
c     ntrunc : order of the truncated expansion
c     Xi     : target vector
c     nt     : number of targets
c     Anm    : precomputed array of scaling coeffs for Pnm
c     Pmax   : dimension parameter for Anm
c---------------------------------------------------------------------
c     OUTPUT:
c     pot    : potential at target (if requested)
c     fld(3) : gradient at target (if requested)
c---------------------------------------------------------------------
      implicit none
      integer i,j,m,n,nt,ier,nterms,ntrunc,Pmax,nbessel
      real *8 r,rx,ry,rz,theta,thetax,thetay,thetaz,scalej
      real *8 phi,phix,phiy,phiz,ctheta,stheta,cphi,sphi,Anm
      real *8 center(3),Xi(3,1),dX(3)
      real *8 Ynm(0:nterms,0:nterms)
      real *8 Ynmd(0:nterms,0:nterms)
      complex *16 wavek,pot(1),fld(3,1)
      complex *16 locexp(0:nterms,-nterms:nterms)
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 jnuse,jn(0:nbessel),jnd(0:nbessel)
      complex *16 imag,ur,utheta,uphi,ztmp,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
      data imag/(0.0d0,1.0d0)/
      ier=0
      do i=1,nt
         dX(1)=Xi(1,i)-center(1)
         dX(2)=Xi(2,i)-center(2)
         dX(3)=Xi(3,i)-center(3)
         call cart2sph(dX,r,theta,phi)
         ctheta = dcos(theta)
         stheta=sqrt(1-ctheta*ctheta)
         cphi=dcos(phi)
         sphi=dsin(phi)
         ephi(0)=1.0d0
         ephi(1)=dcmplx(cphi,sphi)
         ephi(-1)=dconjg(ephi(1))
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi(1)
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
         call ylgndr2sfw(ntrunc,ctheta,Ynm,Ynmd,Anm,Pmax)
         z=wavek*r
         call jfuns3d(ier,ntrunc,z,scalej,jn,1,jnd,
     1	      nbessel)
         pot(i)=pot(i)+locexp(0,0)*jn(0)
         do j=0,ntrunc
            jnd(j)=jnd(j)*wavek
         enddo
         ur = locexp(0,0)*jnd(0)
         utheta = 0.0d0
         uphi = 0.0d0
         do n=1,ntrunc
            pot(i)=pot(i)+locexp(n,0)*jn(n)*Ynm(n,0)
            ur = ur + jnd(n)*Ynm(n,0)*locexp(n,0)
            jnuse = jn(n+1)*scalej + jn(n-1)/scalej
            jnuse = wavek*jnuse/(2*n+1.0d0)
            utheta = utheta -locexp(n,0)*jnuse*Ynmd(n,0)*stheta
            do m=1,n
               ztmp1=jn(n)*Ynm(n,m)*stheta
               ztmp2 = locexp(n,m)*ephi(m)
               ztmp3 = locexp(n,-m)*ephi(-m)
               ztmpsum = ztmp2+ztmp3
               pot(i)=pot(i)+ztmp1*ztmpsum
               ur = ur + jnd(n)*Ynm(n,m)*stheta*ztmpsum
               utheta = utheta -ztmpsum*jnuse*Ynmd(n,m)
               ztmpsum = imag*m*(ztmp2 - ztmp3)
               uphi = uphi + jnuse*Ynm(n,m)*ztmpsum
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

