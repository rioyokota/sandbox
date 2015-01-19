      subroutine P2P(icell,pi,Fi,jcell,Xj,qj,wavek)
      implicit none
      integer i,j,icell(*),jcell(*)
      real*8 R2,R,Xj(3,*),dX(3)
      complex*16 wavek,coef1,coef2,imag/(0.0d0,1.0d0)/
      complex*16 qj(*),pi(*),Fi(3,*)
      do i=icell(8),icell(8)+icell(9)-1
         do j=jcell(8),jcell(8)+jcell(9)-1
            dX(1)=Xj(1,i)-Xj(1,j)
            dX(2)=Xj(2,i)-Xj(2,j)
            dX(3)=Xj(3,i)-Xj(3,j)
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

      subroutine P2M(wavek,scale,Xj,qj,nj,Xi,Mi,Anm1,Anm2)
      use constants, only : P
      implicit none
      integer i,m,n,nj
      real*8 r,theta,phi,ctheta,stheta,scale
      real*8 Xi(3),Xj(3,*),dX(3)
      real*8 Ynm(0:P,0:P)
      real*8 Anm1(0:P,0:P)
      real*8 Anm2(0:P,0:P)
      complex*16 wavek,z,Ynmjn,imag/(0.0d0,1.0d0)/
      complex*16 qj(nj),ephi(P)
      complex*16 jn(0:P+1),jnd(0:P+1)
      complex*16 Mi(0:P,-P:P)
      complex*16 Mnm(0:P,-P:P)
      do n=0,P
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
         do n=2,P
            ephi(n)=ephi(n-1)*ephi(1)
         enddo
         call get_Ynm(P,ctheta,Ynm,Anm1,Anm2)
         z=wavek*r
         call get_jn(P,z,scale,jn,0,jnd)
         do n=0,P
            jn(n)=jn(n)*qj(i)
         enddo
         Mnm(0,0)=Mnm(0,0)+jn(0)
         do n=1,P
            Mnm(n,0)=Mnm(n,0)+Ynm(n,0)*jn(n)
            do m=1,n
               Ynmjn=Ynm(n,m)*jn(n)
               Mnm(n, m)=Mnm(n, m)+Ynmjn*dconjg(ephi(m))
               Mnm(n,-m)=Mnm(n,-m)+Ynmjn*ephi(m)
            enddo
         enddo
      enddo
      do n=0,P
         do m=-n,n
            Mi(n,m)=Mi(n,m)+Mnm(n,m)*imag*wavek
         enddo
      enddo
      return
      end

      subroutine M2M(wavek,scalej,Xj,Mj,scalei,Xi,Mi,
     1     radius,xquad,wquad,nquad,Anm1,Anm2)
      use constants, only : P
      implicit none
      integer l,m,n,mabs,nquad
      real*8 radius,r,theta,phi,ctheta,stheta,cthetaj,rj
      real*8 scalei,scalej
      real*8 Xi(3),Xj(3),dX(3)
      real*8 xquad(2*P),wquad(2*P)
      real*8 ynm(0:P,0:P)
      real*8 Anm1(0:P,0:P)
      real*8 Anm2(0:P,0:P)
      complex*16 wavek,z,imag/(0.0d0,1.0d0)/
      complex*16 Mi(0:P,-P:P)
      complex*16 Mj(0:P,-P:P)
      complex*16 Mnm(0:P,-P:P)
      complex*16 Mrot(0:P,-P:P)
      complex*16 phitemp(nquad,-P:P)
      complex*16 hn(0:P)
      complex*16 ephi(-P:P)
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(1)=exp(imag*phi)
      ephi(0)=1.0d0
      ephi(-1)=dconjg(ephi(1))
      do n=2,P
         ephi(n)=ephi(n-1)*ephi(1)
         ephi(-n)=dconjg(ephi(n))
      enddo
      do n=0,P
         do m=-n,n
            Mnm(n,m)=Mj(n,m)*ephi(m)
         enddo
      enddo
      call rotate(theta,P,Mnm,Mrot)
      do l=1,nquad
         do m=-P,P
            phitemp(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xquad(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         z=wavek*rj
         call get_Ynm(P,cthetaj,ynm,Anm1,Anm2)
         call get_hn(P,z,scalej,hn)
         do m=-P,P
            mabs=abs(m)
            do n=mabs,P
               phitemp(l,m)=phitemp(l,m)+
     1              Mrot(n,m)*hn(n)*ynm(n,mabs)
            enddo
         enddo
      enddo
      do n=0,P
         do m=-n,n
            Mnm(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         call get_Ynm(P,xquad(l),ynm,Anm1,Anm2)
         do m=-P,P
            mabs=abs(m)
            z=phitemp(l,m)*wquad(l)/2
            do n=mabs,P
               Mnm(n,m)=Mnm(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_hn(P,z,scalei,hn)
      do n=0,P
         do m=-n,n
            Mnm(n,m)=Mnm(n,m)/hn(n)
         enddo
      enddo
      call rotate(-theta,P,Mnm,Mrot)
      do n=0,P
         do m=-n,n
            Mnm(n,m)=ephi(-m)*Mrot(n,m)
         enddo
      enddo
      do n=0,min(P,P)
         do m=-n,n
            Mi(n,m)=Mi(n,m)+Mnm(n,m)
         enddo
      enddo
      return
      end

      subroutine M2L(wavek,scalej,Xj,Mj,scalei,Xi,Li,Popt,
     1     radius,xquad,wquad,nquad,Anm1,Anm2)
      use constants, only : P
      implicit none
      integer l,m,n,mabs,Popt,nquad
      real*8 radius,r,theta,phi,ctheta,stheta,cthetaj,sthetaj,thetan
      real*8 rj,rn,scalej,scalei
      real*8 Xi(3),Xj(3),dX(3)
      real*8 xquad(2*P),wquad(2*P)
      real*8 ynm(0:P,0:P),ynmd(0:P,0:P)
      real*8 Anm1(0:P,0:P)
      real*8 Anm2(0:P,0:P)
      complex*16 wavek,z,zh,zhn,ut1,ut2,ut3,imag/(0.0d0,1.0d0)/
      complex*16 phitemp(nquad,-Popt:Popt)
      complex*16 phitempn(nquad,-Popt:Popt)
      complex*16 hn(0:P),hnd(0:P)
      complex*16 jn(0:P+1),jnd(0:P+1)
      complex*16 Mj(0:P,-P:P)
      complex*16 Mnm(0:P,-P:P)
      complex*16 Mrot(0:P,-P:P)
      complex*16 Li(0:P,-P:P)
      complex*16 Lnm(0:P,-P:P)
      complex*16 Lnmd(0:P,-P:P)
      complex*16 Lrot(0:P,-P:P)
      complex*16 ephi(-P-1:P+1)
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(0)=1.0d0
      ephi(1)=exp(imag*phi)
      ephi(-1)=dconjg(ephi(1))
      do n=2,P
         ephi(n)=ephi(n-1)*ephi(1)
         ephi(-n)=dconjg(ephi(n))
      enddo
      do n=0,Popt
         do m=-n,n
            Mnm(n,m)=Mj(n,m)*ephi(m)
         enddo
      enddo
      do n=0,Popt
         do m=-n,n
            Lnm(n,m)=0.0d0
         enddo
      enddo
      call rotate(theta,Popt,Mnm,Mrot)
      do l=1,nquad
         do m=-Popt,Popt
            phitemp(l,m)=0.0d0
            phitempn(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xquad(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         sthetaj=dsqrt(1.0d0-cthetaj**2)
         rn=sthetaj*stheta+cthetaj*ctheta
         thetan=(cthetaj*stheta-ctheta*sthetaj)/rj
         z=wavek*rj
         call get_Ynmd(Popt,cthetaj,ynm,ynmd,Anm1,Anm2)
         call get_hnd(Popt,z,scalej,hn,hnd)
         do n=0,Popt
            hnd(n)=hnd(n)*wavek
         enddo
         do n=1,Popt
            do m=1,n
               ynm(n,m)=ynm(n,m)*sthetaj
            enddo
         enddo
         phitemp(l,0)=Mrot(0,0)*hn(0)
         phitempn(l,0)=Mrot(0,0)*hnd(0)*rn
         do n=1,Popt
            phitemp(l,0)=phitemp(l,0)+Mrot(n,0)*hn(n)*ynm(n,0)
            ut1=hnd(n)*rn
            ut2=hn(n)*thetan
            ut3=ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
            phitempn(l,0)=phitempn(l,0)+ut3*Mrot(n,0)
            do m=1,n
               z=hn(n)*ynm(n,m)
               phitemp(l,m)=phitemp(l,m)+Mrot(n,m)*z
               phitemp(l,-m)=phitemp(l,-m)+Mrot(n,-m)*z
               ut3=ut1*ynm(n,m)-ut2*ynmd(n,m)
               phitempn(l,m)=phitempn(l,m)+ut3*Mrot(n,m)
               phitempn(l,-m)=phitempn(l,-m)+ut3*Mrot(n,-m)
            enddo
         enddo
      enddo
      do n=0,Popt
         do m=-n,n
            Lnm(n,m)=0.0d0
            Lnmd(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         cthetaj=xquad(l)
         call get_Ynm(Popt,cthetaj,ynm,Anm1,Anm2)
         do m=-Popt,Popt
            mabs=abs(m)
            z=phitemp(l,m)*wquad(l)/2.0d0
            do n=mabs,Popt
               Lnm(n,m)=Lnm(n,m)+z*ynm(n,mabs)
            enddo
            z=phitempn(l,m)*wquad(l)/2.0d0
            do n=mabs,Popt
               Lnmd(n,m)=Lnmd(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_jn(Popt,z,scalei,jn,1,jnd)
      do n=0,Popt
         do m=-n,n
            zh=jn(n)
            zhn=jnd(n)*wavek
            z=zh*zh+zhn*zhn
            Lnm(n,m)=(zh*Lnm(n,m)+zhn*Lnmd(n,m))/z
         enddo
      enddo
      call rotate(-theta,Popt,Lnm,Lrot)
      do n=0,Popt
         do m=-n,n
            Lnm(n,m)=ephi(-m)*Lrot(n,m)
         enddo
      enddo
      do n=0,Popt
         do m=-n,n
            Li(n,m)=Li(n,m)+Lnm(n,m)
         enddo
      enddo
      return
      end

      subroutine L2L(wavek,scalej,Xj,Lj,scalei,Xi,Li,
     1     radius,xquad,wquad,nquad,Anm1,Anm2)
      use constants, only : P
      implicit none
      integer l,m,n,mabs,nquad
      real*8 radius,r,theta,phi,ctheta,stheta,cthetaj,sthetaj,thetan
      real*8 rj,rn,scalej,scalei
      real*8 Xi(3),Xj(3),dX(3)
      real*8 xquad(2*P),wquad(2*P)
      real*8 ynm(0:P,0:P),ynmd(0:P,0:P)
      real*8 Anm1(0:P,0:P)
      real*8 Anm2(0:P,0:P)
      complex*16 wavek,z,zh,zhn,ut1,ut2,ut3,imag/(0.0d0,1.0d0)/
      complex*16 phitemp(nquad,-P:P)
      complex*16 phitempn(nquad,-P:P)
      complex*16 jn(0:P+1)
      complex*16 jnd(0:P+1)
      complex*16 Li(0:P,-P:P)
      complex*16 Lj(0:P,-P:P)
      complex*16 Lnm(0:P,-P:P)
      complex*16 Lnmd(0:P,-P:P)
      complex*16 Lrot(0:P,-P:P)
      complex*16 ephi(-P-1:P+1)
      dX(1)=Xi(1)-Xj(1)
      dX(2)=Xi(2)-Xj(2)
      dX(3)=Xi(3)-Xj(3)
      call cart2sph(dX,r,theta,phi)
      ephi(0)=1.0d0
      ephi(1)=exp(imag*phi)
      ephi(-1)=dconjg(ephi(1))
      do n=1,P
         ephi(n+1)=ephi(n)*ephi(1)
         ephi(-1-n)=dconjg(ephi(n+1))
      enddo
      do n=0,P
         do m=-n,n
            Lnm(n,m)=Lj(n,m)*ephi(m)
         enddo
      enddo
      call rotate(theta,P,Lnm,Lrot)
      do n=0,P
         do m=-n,n
            Lnm(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         do m=-P,P
            phitemp(l,m)=0.0d0
            phitempn(l,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         ctheta=xquad(l)
         stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
         cthetaj=(r+radius*ctheta)/rj
         sthetaj=dsqrt(1.0d0-cthetaj**2)
         rn=sthetaj*stheta+cthetaj*ctheta
         thetan=(cthetaj*stheta-sthetaj*ctheta)/rj
         z=wavek*rj
         call get_Ynmd(P,cthetaj,ynm,ynmd,Anm1,Anm2)
         call get_jn(P,z,scalej,jn,1,jnd)
         do n=0,P
            jnd(n)=jnd(n)*wavek
         enddo
         do n=1,P
            do m=1,n
               ynm(n,m)=ynm(n,m)*sthetaj
            enddo
         enddo
         phitemp(l,0)=Lrot(0,0)*jn(0)
         phitempn(l,0)=Lrot(0,0)*jnd(0)*rn
         do n=1,P
            phitemp(l,0)=phitemp(l,0)+Lrot(n,0)*jn(n)*ynm(n,0)
            ut1=jnd(n)*rn
            ut2=jn(n)*thetan
            ut3=ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
            phitempn(l,0)=phitempn(l,0)+ut3*Lrot(n,0)
            do m=1,min(n,P)
               z=jn(n)*ynm(n,m)
               phitemp(l,m)=phitemp(l,m)+Lrot(n,m)*z
               phitemp(l,-m)=phitemp(l,-m)+Lrot(n,-m)*z
               ut3=ut1*ynm(n,m)-ut2*ynmd(n,m)
               phitempn(l,m)=phitempn(l,m)+ut3*Lrot(n,m)
               phitempn(l,-m)=phitempn(l,-m)+ut3*Lrot(n,-m)
            enddo
         enddo
      enddo
      do n=0,P
         do m=-n,n
            Lnm(n,m)=0.0d0
            Lnmd(n,m)=0.0d0
         enddo
      enddo
      do l=1,nquad
         cthetaj=xquad(l)
         call get_Ynm(P,cthetaj,ynm,Anm1,Anm2)
         do m=-P,P
            mabs=abs(m)
            z=phitemp(l,m)*wquad(l)/2.0d0
            do n=mabs,P
               Lnm(n,m)=Lnm(n,m)+z*ynm(n,mabs)
            enddo
            z=phitempn(l,m)*wquad(l)/2.0d0
            do n=mabs,P
               Lnmd(n,m)=Lnmd(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      z=wavek*radius
      call get_jn(P,z,scalei,jn,1,jnd)
      do n=0,P
         do m=-n,n
            zh=jn(n)
            zhn=jnd(n)*wavek
            z=zh*zh+zhn*zhn
            Lnm(n,m)=(zh*Lnm(n,m)+zhn*Lnmd(n,m))/z
         enddo
      enddo
      call rotate(-theta,P,Lnm,Lrot)
      do n=0,P
         do m=-n,n
            Lnm(n,m)=ephi(-m)*Lrot(n,m)
         enddo
      enddo
      do n=0,P
         do m=-n,n
            Li(n,m)=Li(n,m)+Lnm(n,m)
         enddo
      enddo
      return
      end

      subroutine L2P(wavek,scalej,Xj,Lj,
     1     Xi,ni,pi,Fi,Anm1,Anm2)
      use constants, only : P
      implicit none
      integer i,j,m,n,ni
      real*8 r,rx,ry,rz,theta,thetax,thetay,thetaz,scalej
      real*8 phi,phix,phiy,phiz,ctheta,stheta,cphi,sphi
      real*8 Xj(3),Xi(3,1),dX(3)
      real*8 Ynm(0:P,0:P)
      real*8 Ynmd(0:P,0:P)
      real*8 Anm1(0:P,0:P)
      real*8 Anm2(0:P,0:P)
      complex*16 wavek,imag/(0.0d0,1.0d0)/
      complex*16 ur,utheta,uphi,z
      complex*16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex*16 ux,uy,uz
      complex*16 pi(1),Fi(3,1)
      complex*16 Lj(0:P,-P:P)
      complex*16 ephi(P+1)
      complex*16 jnuse,jn(0:P+1),jnd(0:P+1)
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
         do j=2,P+1
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
         call get_Ynmd(P,ctheta,Ynm,Ynmd,Anm1,Anm2)
         z=wavek*r
         call get_jn(P,z,scalej,jn,1,jnd)
         pi(i)=pi(i)+Lj(0,0)*jn(0)
         do j=0,P
            jnd(j)=jnd(j)*wavek
         enddo
         ur=Lj(0,0)*jnd(0)
         utheta=0.0d0
         uphi=0.0d0
         do n=1,P
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

