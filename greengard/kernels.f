c**********************************************************************
      subroutine P2P(ibox,Xi,pi,Fi,jbox,Xj,qj,wavek)
c**********************************************************************
c     This subroutine calculates the potential and field
c     at the target point Xi, due to a charge at Xj.
c     The scaling is that required of the delta function response: i.e.,
c     pi = exp(i*k*r)/r
c     Fi = -grad(pi)
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
      complex *16 wavek,coef1,coef2,imag/(0.0d0,1.0d0)/
      complex *16 qj(1000000),pi(1000000),Fi(3,1000000)
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
      subroutine P2M(wavek,scale,Xj,qj,nj,Xi,
     1     nterms,ntrunc,nbessel,Mi,Anm1,Anm2,Pmax)
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
      integer i,m,n,nj,nterms,ntrunc,Pmax,nbessel
      real *8 r,theta,phi,ctheta,stheta,scale
      real *8 Xi(3),Xj(3,nj),dX(3)
      real *8 Ynm(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek,z,Ynmjn,imag/(0.0d0,1.0d0)/
      complex *16 qj(nj),ephi(ntrunc)
      complex *16 jn(0:nbessel),jnd(0:nbessel)
      complex *16 Mi(0:nterms,-nterms:nterms)
      complex *16 Mnm(0:nterms,-nterms:nterms)
      do n=0,nterms
         do m=-n,n
            Mnm(n,m)=0
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
         call get_Ynm(ntrunc,ctheta,Ynm,Anm1,Anm2,Pmax)
         z=wavek*r
         call get_jn(ntrunc,z,scale,jn,0,jnd,nbessel)
         do n=0,ntrunc
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
     2     radius,xnodes,wts,nquad,Anm1,Anm2,Pmax)
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
      integer l,m,n,mabs,ntermsi,ntermsj,nquad,Pmax
      real *8 radius,r,theta,phi,ctheta,stheta,cthetaj,rj
      real *8 scalei,scalej
      real *8 Xi(3),Xj(3),dX(3)
      real *8 xnodes(nquad),wts(nquad)
      real *8 ynm(0:ntermsj,0:ntermsj)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek,z,imag/(0.0d0,1.0d0)/
      complex *16 Mi(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Mj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 phitemp(nquad,-ntermsj:ntermsj)
      complex *16 fhs(0:ntermsj)
      complex *16 ephi(-ntermsj-1:ntermsj+1)
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(1)=exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do n=1,ntermsj
         ephi(n+1)=ephi(n)*ephi(1)
         ephi(-1-n)=dconjg(ephi(n+1))
      enddo
      do n=0,ntermsj
         do m=-n,n
            Mnm(n,m)=Mj(n,m)*ephi(m)
         enddo
      enddo
      call rotate(theta,ntermsj,Mnm,ntermsj,Mrot)
      do l=1,nquad
         do m=-ntermsj,ntermsj
            phitemp(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xnodes(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         z=wavek*rj
         call get_Ynm(ntermsj,cthetaj,ynm,Anm1,Anm2,Pmax)
         call get_hn(ntermsj,z,scalej,fhs)
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
      do l=1,nquad
         call get_Ynm(ntermsi,xnodes(l),ynm,Anm1,Anm2,Pmax)
         do m=-ntermsj,ntermsj
            mabs=abs(m)
            z=phitemp(l,m)*wts(l)/2
            do n=mabs,ntermsi
               Mnm(n,m)=Mnm(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_hn(ntermsi,z,scalei,fhs)
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
      do n=0,min(ntermsj,ntermsi)
         do m=-n,n
            Mi(n,m)=Mi(n,m)+Mnm(n,m)
         enddo
      enddo
      return
      end
c***********************************************************************
      subroutine M2L(wavek,scalej,Xj,Mj,
     1     ntermsj,scalei,Xi,Li,ntermsi,ntrunc,
     1     radius,xnodes,wts,nquad,nbessel,Anm1,Anm2,Pmax)
c***********************************************************************
c     Convert multipole expansion to a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     Xj      : center of original multiple expansion
c     Xi      : center of shifted local expansion
c     Mj      : coefficients of original multiple expansion
c     ntermsj : order of multipole expansion
c     scalej  : scaling parameter for mpole expansion
c     scalei  : scaling parameter for local expansion
c     radius  : radius of sphere on which local expansion is computed
c     xnodes  : Legendre nodes (precomputed)
c     wts     : Legendre weights (precomputed)
c     nquad   : number of quadrature nodes used (really nquad**2)
c---------------------------------------------------------------------
c     OUTPUT:
c     Li      : coefficients of shifted local expansion
c---------------------------------------------------------------------
      implicit none
      integer l,m,n,mabs,ntermsi,ntermsj,ntrunc,nquad,nbessel,Pmax
      real *8 radius,r,theta,phi,ctheta,stheta,cthetaj,sthetaj,thetan
      real *8 rj,rn,scalej,scalei
      real *8 Xi(3),Xj(3),dX(3)
      real *8 xnodes(nquad),wts(nquad)
      real *8 ynm(0:ntrunc,0:ntrunc),ynmd(0:ntrunc,0:ntrunc)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek,z,zh,zhn,ut1,ut2,ut3,imag/(0.0d0,1.0d0)/
      complex *16 phitemp(nquad,-ntrunc:ntrunc)
      complex *16 phitempn(nquad,-ntrunc:ntrunc)
      complex *16 fhs(0:ntrunc),fhder(0:ntrunc)
      complex *16 jn(0:nbessel),jnd(0:nbessel)
      complex *16 Mj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Mrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Li(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Lnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Lnmd(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Lrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 ephi(-ntermsj-1:ntermsj+1)
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
            Mnm(n,m)=Mj(n,m)*ephi(m)
         enddo
      enddo
      do n=0,ntrunc
         do m=-n,n
            Lnm(n,m)=0.0d0
         enddo
      enddo
      call rotate(theta,ntrunc,Mnm,ntermsj,Mrot)
      do l=1,nquad
         do m=-ntrunc,ntrunc
            phitemp(l,m)=0.0d0
            phitempn(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xnodes(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         sthetaj=dsqrt(1.0d0-cthetaj**2)
         rn=sthetaj*stheta+cthetaj*ctheta
         thetan=(cthetaj*stheta-ctheta*sthetaj)/rj
         z=wavek*rj
         call get_Ynmd(ntrunc,cthetaj,ynm,ynmd,Anm1,Anm2,Pmax)
         call get_hnd(ntrunc,z,scalej,fhs,fhder)
         do n=0,ntrunc
            fhder(n)=fhder(n)*wavek
         enddo
         do n=1,ntrunc
            do m=1,n
               ynm(n,m)=ynm(n,m)*sthetaj
            enddo
         enddo
         phitemp(l,0)=Mrot(0,0)*fhs(0)
         phitempn(l,0)=Mrot(0,0)*fhder(0)*rn
         do n=1,ntrunc
            phitemp(l,0)=phitemp(l,0)+Mrot(n,0)*fhs(n)*ynm(n,0)
            ut1=fhder(n)*rn
            ut2=fhs(n)*thetan
            ut3=ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
            phitempn(l,0)=phitempn(l,0)+ut3*Mrot(n,0)
            do m=1,n
               z=fhs(n)*ynm(n,m)
               phitemp(l,m)=phitemp(l,m)+Mrot(n,m)*z
               phitemp(l,-m)=phitemp(l,-m)+Mrot(n,-m)*z
               ut3=ut1*ynm(n,m)-ut2*ynmd(n,m)
               phitempn(l,m)=phitempn(l,m)+ut3*Mrot(n,m)
               phitempn(l,-m)=phitempn(l,-m)+ut3*Mrot(n,-m)
            enddo
         enddo
      enddo
      do n=0,ntrunc
         do m=-n,n
            Lnm(n,m)=0.0d0
            Lnmd(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         cthetaj=xnodes(l)
         call get_Ynm(ntrunc,cthetaj,ynm,Anm1,Anm2,Pmax)
         do m=-ntrunc,ntrunc
            mabs=abs(m)
            z=phitemp(l,m)*wts(l)/2.0d0
            do n=mabs,ntrunc
               Lnm(n,m)=Lnm(n,m)+z*ynm(n,mabs)
            enddo
            z=phitempn(l,m)*wts(l)/2.0d0
            do n=mabs,ntrunc
               Lnmd(n,m)=Lnmd(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_jn(ntrunc,z,scalei,jn,1,jnd,nbessel)
      do n=0,ntrunc
         do m=-n,n
            zh=jn(n)
            zhn=jnd(n)*wavek
            z=zh*zh+zhn*zhn
            Lnm(n,m)=(zh*Lnm(n,m)+zhn*Lnmd(n,m))/z
         enddo
      enddo
      call rotate(-theta,ntrunc,Lnm,ntermsj,Lrot)
      do n=0,ntrunc
         do m=-n,n
            Lnm(n,m)=ephi(-m)*Lrot(n,m)
         enddo
      enddo
      do n=0,ntrunc
         do m=-n,n
            Li(n,m)=Li(n,m)+Lnm(n,m)
         enddo
      enddo
      return
      end
      subroutine L2L(wavek,scalej,Xj,Lj,ntermsj,
     1     scalei,Xi,Li,ntermsi,
     2     radius,xnodes,wts,nquad,nbessel,Anm1,Anm2,Pmax)
c***********************************************************************
c     Shifts center of a local expansion.
c     This is a reasonably fast "point and shoot" version which
c     first rotates the coordinate system, then doing the shifting
c     along the Z-axis, and then rotating back to the original
c     coordinates.
c---------------------------------------------------------------------
c     INPUT:
c     wavek   : Helmholtz parameter
c     Xi      : center of shifted local expansion
c     Xj      : center of original multiple expansion
c     Lj      : coefficients of original multiple expansion
c     scalei  : scaling parameter for local expansion
c     scalej  : scaling parameter for local expansion
c     ntermsi : order of new local expansion
c     ntermsj : order of original local expansion
c     marray  : work array
c     dc      : another work array
c     rd1     : work array for rotation operators.
c     rd2     : work array for rotation operators.
c     ephi    : work array for rotation operators.
c     radius  : radius of sphere on which local expansion is computed
c     xnodes  : Legendre nodes (precomputed)
c     wts     : Legendre weights (precomputed)
c     nquad   : number of quadrature nodes in theta direction.
c---------------------------------------------------------------------
c     OUTPUT:
c     Li      : coefficients of shifted local expansion
c***********************************************************************
      implicit none
      integer l,m,n,mabs,ntermsi,ntermsj,nquad,nbessel,Pmax
      real *8 radius,r,theta,phi,ctheta,stheta,cthetaj,sthetaj,thetan
      real *8 rj,rn,scalej,scalei
      real *8 Xi(3),Xj(3),dX(3)
      real *8 xnodes(nquad),wts(nquad)
      real *8 ynm(0:ntermsi,0:ntermsi),ynmd(0:ntermsi,0:ntermsi)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek,z,zh,zhn,ut1,ut2,ut3,imag/(0.0d0,1.0d0)/
      complex *16 phitemp(nquad,-ntermsi:ntermsi)
      complex *16 phitempn(nquad,-ntermsi:ntermsi)
      complex *16 jn(0:nbessel)
      complex *16 jnd(0:nbessel)
      complex *16 Li(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Lj(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Lnm(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Lnmd(0:ntermsi,-ntermsi:ntermsi)
      complex *16 Lrot(0:ntermsi,-ntermsi:ntermsi)
      complex *16 ephi(-ntermsi-1:ntermsi+1)
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(0)=1.0d0
      ephi(1)=exp(imag*phi)
      ephi(-1)=dconjg(ephi(1))
      do n=1,ntermsi
         ephi(n+1)=ephi(n)*ephi(1)
         ephi(-1-n)=dconjg(ephi(n+1))
      enddo
      do n=0,ntermsj
         do m=-n,n
            Lnm(n,m)=Lj(n,m)*ephi(m)
         enddo
      enddo
      call rotate(theta,ntermsj,Lnm,ntermsi,Lrot)
      do n=0,ntermsi
         do m=-n,n
            Lnm(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         do m=-ntermsi,ntermsi
            phitemp(l,m)=0.0d0
            phitempn(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xnodes(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         sthetaj=dsqrt(1.0d0-cthetaj**2)
         rn=sthetaj*stheta+cthetaj*ctheta
         thetan=(cthetaj*stheta-sthetaj*ctheta)/rj
         z=wavek*rj
         call get_Ynmd(ntermsj,cthetaj,ynm,ynmd,Anm1,Anm2,Pmax)
         call get_jn(ntermsj,z,scalej,jn,1,jnd,nbessel)
         do n=0,ntermsj
            jnd(n)=jnd(n)*wavek
         enddo
         do n=1,ntermsj
            do m=1,n
               ynm(n,m)=ynm(n,m)*sthetaj
            enddo
         enddo
         phitemp(l,0)=Lrot(0,0)*jn(0)
         phitempn(l,0)=Lrot(0,0)*jnd(0)*rn
         do n=1,ntermsj
            phitemp(l,0)=phitemp(l,0)+Lrot(n,0)*jn(n)*ynm(n,0)
            ut1=jnd(n)*rn
            ut2=jn(n)*thetan
            ut3=ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
            phitempn(l,0)=phitempn(l,0)+ut3*Lrot(n,0)
            do m=1,min(n,ntermsi)
               z=jn(n)*ynm(n,m)
               phitemp(l,m)=phitemp(l,m)+Lrot(n,m)*z
               phitemp(l,-m)=phitemp(l,-m)+Lrot(n,-m)*z
               ut3=ut1*ynm(n,m)-ut2*ynmd(n,m)
               phitempn(l,m)=phitempn(l,m)+ut3*Lrot(n,m)
               phitempn(l,-m)=phitempn(l,-m)+ut3*Lrot(n,-m)
            enddo
         enddo
      enddo
      do n=0,ntermsi
         do m=-n,n
            Lnm(n,m)=0.0d0
            Lnmd(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         cthetaj=xnodes(l)
         call get_Ynm(ntermsi,cthetaj,ynm,Anm1,Anm2,Pmax)
         do m=-ntermsi,ntermsi
            mabs=abs(m)
            z=phitemp(l,m)*wts(l)/2.0d0
            do n=mabs,ntermsi
               Lnm(n,m)=Lnm(n,m)+z*ynm(n,mabs)
            enddo
            z=phitempn(l,m)*wts(l)/2.0d0
            do n=mabs,ntermsi
               Lnmd(n,m)=Lnmd(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_jn(ntermsi,z,scalei,jn,1,jnd,nbessel)
      do n=0,ntermsi
         do m=-n,n
            zh=jn(n)
            zhn=jnd(n)*wavek
            z=zh*zh+zhn*zhn
            Lnm(n,m)=(zh*Lnm(n,m)+zhn*Lnmd(n,m))/z
         enddo
      enddo
      call rotate(-theta,ntermsi,Lnm,ntermsi,Lrot)
      do n=0,ntermsi
         do m=-n,n
            Lnm(n,m)=ephi(-m)*Lrot(n,m)
         enddo
      enddo
      do n=0,ntermsi
         do m=-n,n
            Li(n,m)=Li(n,m)+Lnm(n,m)
         enddo
      enddo
      return
      end
c**********************************************************************
      subroutine L2P(wavek,scalej,Xj,Lj,nterms,
     1     ntrunc,nbessel,Xi,ni,pi,Fi,Anm1,Anm2,Pmax)
c**********************************************************************
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point TARGET.
c     pi= sum sum  Lj(n,m) j_n(k r) Y_nm(theta,phi)
c     n   m
c---------------------------------------------------------------------
c     INPUT:
c     wavek  : the Helmholtz coefficient
c     scalej : scaling parameter used in forming expansion
c     Xj     : coordinates of the expansion center
c     Lj     : coeffs of the j-expansion
c     nterms : order of the h-expansion
c     ntrunc : order of the truncated expansion
c     Xi     : target vector
c     ni     : number of targets
c     Anm    : precomputed array of scaling coeffs for Pnm
c     Pmax   : dimension parameter for Anm
c---------------------------------------------------------------------
c     OUTPUT:
c     pi     : potential at target (if requested)
c     Fi     : gradient at target (if requested)
c---------------------------------------------------------------------
      implicit none
      integer i,j,m,n,ni,nterms,ntrunc,Pmax,nbessel
      real *8 r,rx,ry,rz,theta,thetax,thetay,thetaz,scalej
      real *8 phi,phix,phiy,phiz,ctheta,stheta,cphi,sphi
      real *8 Xj(3),Xi(3,1),dX(3)
      real *8 Ynm(0:nterms,0:nterms)
      real *8 Ynmd(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax)
      real *8 Anm2(0:Pmax,0:Pmax)
      complex *16 wavek,imag/(0.0d0,1.0d0)/
      complex *16 ur,utheta,uphi,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
      complex *16 pi(1),Fi(3,1)
      complex *16 Lj(0:nterms,-nterms:nterms)
      complex *16 ephi(nterms)
      complex *16 jnuse,jn(0:nbessel),jnd(0:nbessel)
      do i=1,ni
         dX(1)=Xi(1,i)-Xj(1)
         dX(2)=Xi(2,i)-Xj(2)
         dX(3)=Xi(3,i)-Xj(3)
         call cart2sph(dX,r,theta,phi)
         ctheta=dcos(theta)
         stheta=sqrt(1-ctheta*ctheta)
         cphi=dcos(phi)
         sphi=dsin(phi)
         ephi(1)=dcmplx(cphi,sphi)
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi(1)
         enddo
         rx=stheta*cphi
         thetax=ctheta*cphi
         phix=-sphi
         ry=stheta*sphi
         thetay=ctheta*sphi
         phiy=cphi
         rz=ctheta
         thetaz=-stheta
         phiz=0.0d0
         call get_Ynmd(ntrunc,ctheta,Ynm,Ynmd,Anm1,Anm2,Pmax)
         z=wavek*r
         call get_jn(ntrunc,z,scalej,jn,1,jnd,nbessel)
         pi(i)=pi(i)+Lj(0,0)*jn(0)
         do j=0,ntrunc
            jnd(j)=jnd(j)*wavek
         enddo
         ur=Lj(0,0)*jnd(0)
         utheta=0.0d0
         uphi=0.0d0
         do n=1,ntrunc
            pi(i)=pi(i)+Lj(n,0)*jn(n)*Ynm(n,0)
            ur=ur+jnd(n)*Ynm(n,0)*Lj(n,0)
            jnuse=jn(n+1)*scalej+jn(n-1)/scalej
            jnuse=wavek*jnuse/(2*n+1.0d0)
            utheta=utheta-Lj(n,0)*jnuse*Ynmd(n,0)*stheta
            do m=1,n
               ztmp1=jn(n)*Ynm(n,m)*stheta
               ztmp2=Lj(n,m)*ephi(m)
               ztmp3=Lj(n,-m)*dconjg(ephi(m))
               ztmpsum=ztmp2+ztmp3
               pi(i)=pi(i)+ztmp1*ztmpsum
               ur=ur+jnd(n)*Ynm(n,m)*stheta*ztmpsum
               utheta=utheta-ztmpsum*jnuse*Ynmd(n,m)
               ztmpsum=imag*m*(ztmp2-ztmp3)
               uphi=uphi+jnuse*Ynm(n,m)*ztmpsum
            enddo
         enddo
         ux=ur*rx+utheta*thetax+uphi*phix
         uy=ur*ry+utheta*thetay+uphi*phiy
         uz=ur*rz+utheta*thetaz+uphi*phiz
         Fi(1,i)=Fi(1,i)-ux
         Fi(2,i)=Fi(2,i)-uy
         Fi(3,i)=Fi(3,i)-uz
      enddo
      return
      end

