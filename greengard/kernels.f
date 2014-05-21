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
      real *8 R2,R,Xi(3,1000000),Xj(3,1000000),dX(3)
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
      integer l,m,n,mabs,ntermsi,ntermsj,nquad,ier
      real *8 radius,r,theta,phi,ctheta,stheta,cthetaj,rj
      real *8 scalei,scalej
      real *8 Xi(3),Xj(3),dX(3)
      real *8 xnodes(nquad),wts(nquad)
      real *8 ynm(0:ntermsj,0:ntermsj)
      complex *16 Mi(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Mj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 phitemp(nquad,-ntermsj:ntermsj)
      complex *16 fhs(0:ntermsj),fhder(0:ntermsj)
      real *8 rat1(0:ntermsj,0:ntermsj),rat2(0:ntermsj,0:ntermsj)
      complex *16 imag,wavek,z
      complex *16 ephi(-ntermsj-1:ntermsj+1)
      data imag/(0.0d0,1.0d0)/
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(1) = exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do n=1,ntermsj
         ephi(n+1) = ephi(n)*ephi(1)
         ephi(-1-n) = dconjg(ephi(n+1))
      enddo
      do n=0,ntermsj
         do m=-n,n
            Mnm(n,m)=Mj(n,m)*ephi(m)
         enddo
      enddo
      call rotate(theta,ntermsj,Mnm,ntermsj,Mrot)
      do n=0,ntermsi
         do m=-n,n
            Mnm(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         do m=-ntermsj,ntermsj
            phitemp(l,m) = 0.0d0
         enddo
      enddo
      call ylgndrini(ntermsj,rat1,rat2)
      do l=1,nquad
         ctheta=xnodes(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         z=wavek*rj
         call ylgndrf(ntermsj,cthetaj,ynm,rat1,rat2)
         call h3dall(ntermsj,z,scalej,fhs,0,fhder)
         do m=-ntermsj,ntermsj
            mabs=abs(m)
            do n=mabs,ntermsj
               phitemp(l,m)=phitemp(l,m)+
     1              Mrot(n,m)*fhs(n)*ynm(n,mabs)
            enddo
         enddo
      enddo
      do n=0,ntermsi
         do m=-n,n
            Mnm(n,m)=0.0d0
         enddo
      enddo
      call ylgndrini(ntermsi,rat1,rat2)
      do l=1,nquad
         call ylgndrf(ntermsi,xnodes(l),ynm,rat1,rat2)
         do m=-ntermsj,ntermsj
            mabs=abs(m)
            z=phitemp(l,m)*wts(l)/2
            do n=mabs,ntermsi
               Mnm(n,m)=Mnm(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z = wavek*radius
      call h3dall(ntermsi,z,scalei,fhs,0,fhder)
      do n=0,ntermsi
         do m=-n,n
            Mnm(n,m)=Mnm(n,m)/fhs(n)
         enddo
      enddo
      call rotate(-theta,ntermsi,Mnm,ntermsj,Mrot)
      do n=0,ntermsi
         do m=-n,n
            Mnm(n,m)=ephi(-m)*Mrot(n,m)
         enddo
      enddo
      do n = 0,min(ntermsj,ntermsi)
         do m=-n,n
            Mi(n,m) = Mi(n,m)+Mnm(n,m)
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2L(wavek,scalej,Xj,Mj,
     1     ntermsj,scalei,Xi,Li,ntermsi,ntrunc,
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
c     Li : coefficients of shifted local expansion
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer ntermsj,ier,l,m,jnew,knew
      real *8 r,theta,ctheta,phi,scalej,scalei
      real *8 Xj(3),Xi(3)
      real *8 xnodes(1),wts(1),dX(3)
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
      complex *16 Mnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Mrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Li(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Lnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Lrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 imag,wavek
      complex *16 ephi(-ntermsj-1:ntermsj+1)
      data imag/(0.0d0,1.0d0)/
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(0)=1.0d0
      ephi(1)=exp(imag*phi)
      ephi(-1)=dconjg(ephi(1))
      do n=1,ntermsj
         ephi(n+1)=ephi(n)*ephi(1)
         ephi(-1-n)=dconjg(ephi(n+1))
      enddo
      do n=0,ntrunc
         do m=-n,n
            Mnm(n,m) = Mj(n,m)*ephi(m)
         enddo
      enddo
      do n=0,ntrunc
         do m=-n,n
            Lnm(n,m)=0.0d0
         enddo
      enddo
      call rotate(theta,ntrunc,Mnm,ntermsj,Mrot)
      call h3dmpevalspherenmstab_fast(Mrot,wavek,scalej,r,radius,
     2     ntrunc,ntermsj,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     3     fhs,fhder)
      call h3dprojlocsepstab_fast
     $   (ntrunc,ntrunc,nquad,ntrunc,xnodes,wts,
     1     phitemp,phitempn,Lnm,mp2,ynm)
      call h3drescalestab(ntrunc,ntrunc,Lnm,mp2,
     1     radius,wavek,scalei,jn,jnd,nbessel,ier)
      call rotate(-theta,ntrunc,Lnm,ntermsj,Lrot)
      do l=0,ntrunc
         do m=-l,l
            Lnm(l,m) = ephi(-m)*Lrot(l,m)
         enddo
      enddo
      do l = 0,ntrunc
         do m=-l,l
            Li(l,m) = Li(l,m)+Lnm(l,m)
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
      real *8 r,theta,ctheta,phi,scalej,scalei
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
      call cart2sph(dX,r,theta,phi)
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
      call rotate(theta,ntermsj,marray1,ldc,marray)
      call h3dlocevalspherestab_fast(marray,wavek,scalej,
     1     r,radius,ntermsj,ntermsi,
     1     ldc,ynm,ynmd,phitemp,phitempn,nquad,xnodes,
     1     jn,jnd,nbessel,ier)
      call h3dprojlocsepstab_fast
     1     (ntermsi,ntermsi,nquad,ntermsi,xnodes,wts,
     1     phitemp,phitempn,mptemp,mp2,ynm)
      call h3drescalestab(ntermsi,ntermsi,mptemp,mp2,
     1      radius,wavek,scalei,jn,jnd,nbessel,ier)
      call rotate(-theta,ntermsi,mptemp,ldc,marray)
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

