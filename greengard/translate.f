      subroutine cart2sph(dX, r, theta, phi)
      implicit none
      real *8 r,theta,phi,dX(3)
      r = sqrt(dX(1) * dX(1) + dX(2) * dX(2) + dX(3) * dX(3))
      theta = datan2(sqrt(dX(1) * dX(1) + dX(2) * dX(2)), dX(3))
      if(abs(dX(1)).eq.0.and.abs(dX(2)).eq.0) then
         phi = 0
      else
         phi = datan2(dX(2), dX(1))
      endif
      return
      end

      subroutine initCoefs(C, nterms)
      implicit none
      integer nterms,m,n
      complex *16 C(0:nterms,-nterms:nterms)
      do n=0,nterms
         do m=-n,n
            C(n,m) = 0
         enddo
      enddo
      return
      end

      subroutine getAnm(Pmax, Anm1, Anm2)
      implicit none
      integer Pmax,m,n
      real *8 Anm1(0:Pmax,0:Pmax),Anm2(0:Pmax,0:Pmax)
      Anm1(0,0) = 1
      Anm2(0,0) = 1
      do m=0,Pmax
         if (m.gt.0) Anm1(m,m) = sqrt((2 * m - 1.0d0) / (2 * m))
         if (m.lt.Pmax) Anm1(m+1,m) = sqrt(2 * m + 1.0d0)
         do n = m+2, Pmax
            Anm1(n,m) = (2*n-1)
            Anm2(n,m) = sqrt((n + m - 1.0d0) * (n - m - 1.0d0))
            Anm1(n,m) = Anm1(n,m) / sqrt(dble(n - m) * (n + m))
            Anm2(n,m) = Anm2(n,m) / sqrt(dble(n - m) * (n + m))
         enddo
      enddo
      return
      end

      subroutine rotate(theta,ntermsj,Mnm,ntermsi,Mrot)
      implicit none
      integer ntermsi,ntermsj,n,m,mp
      real *8 theta,ctheta,stheta,hsthta,cthtap,cthtan,d
      real *8 scale,eps/1.0d-15/
      real *8 Rnm1(0:ntermsj,-ntermsj:ntermsj)
      real *8 Rnm2(0:ntermsj,-ntermsj:ntermsj)
      real *8 sqrtCnm(0:2*ntermsj,2)
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsi,-ntermsi:ntermsi)
      do m=0,2*ntermsj
         sqrtCnm(m,1)=dsqrt(m+0.0d0)
      enddo
      sqrtCnm(0,2)=0.0d0
      if(ntermsj.gt.0) sqrtCnm(1,2)=0.0d0
      do m=2,2*ntermsj
         sqrtCnm(m,2)=dsqrt((m+0.0d0)*(m-1)/2.0d0)
      enddo
      ctheta=dcos(theta)
      if(dabs(ctheta).le.eps) ctheta=0.0d0
      stheta=dsin(-theta)
      if(dabs(stheta).le.eps) stheta=0.0d0
      hsthta=stheta/sqrt(2.0d0)
      cthtap= sqrt(2.0d0)*dcos(theta/2.0d0)**2
      cthtan=-sqrt(2.0d0)*dsin(theta/2.0d0)**2
      Rnm1(0,0)=1.0d0
      Mrot(0,0)=Mnm(0,0)*Rnm1(0,0)
      do n=1,ntermsj
         do m=-n,-1
            Rnm2(0,m)=-sqrtCnm(n-m,2)*Rnm1(0,m+1)
            if (m.gt.(1-n)) then
               Rnm2(0,m)=Rnm2(0,m)+sqrtCnm(n+m,2)*Rnm1(0,m-1)
            endif
            Rnm2(0,m)=Rnm2(0,m)*hsthta
            if (m.gt.-n) then
               Rnm2(0,m)=Rnm2(0,m)+
     $              Rnm1(0,m)*ctheta*sqrtCnm(n+m,1)*sqrtCnm(n-m,1)
            endif
            Rnm2(0,m)=Rnm2(0,m)/n
         enddo
         Rnm2(0,0)=Rnm1(0,0)*ctheta
         if (n.gt.1) then
            Rnm2(0,0)=Rnm2(0,0)+hsthta*sqrtCnm(n,2)*(2*Rnm1(0,-1))/n
         endif
         do m=1,n
            Rnm2(0,m)=Rnm2(0,-m)
            if(mod(m,2).eq.0) then
               Rnm2(m,0)=+Rnm2(0,m)
            else
               Rnm2(m,0)=-Rnm2(0,m)
            endif
         enddo
         do mp=1,n
            scale=1/(sqrt(2.0d0)*sqrtCnm(n+mp,2))
            do m=mp,n
               Rnm2(mp,+m)=Rnm1(mp-1,+m-1)*(cthtap*sqrtCnm(n+m,2))
               Rnm2(mp,-m)=Rnm1(mp-1,-m+1)*(cthtan*sqrtCnm(n+m,2))
               if (m.lt.(n-1)) then
                  Rnm2(mp,+m)=Rnm2(mp,+m)-Rnm1(mp-1,+m+1)*
     $                 (cthtan*sqrtCnm(n-m,2))
                  Rnm2(mp,-m)=Rnm2(mp,-m)-Rnm1(mp-1,-m-1)*
     $                 (cthtap*sqrtCnm(n-m,2))
               endif
               if (m.lt.n) then
                  d=(stheta*sqrtCnm(n+m,1)*sqrtCnm(n-m,1))
                  Rnm2(mp,+m)=Rnm2(mp,+m)+Rnm1(mp-1,+m)*d
                  Rnm2(mp,-m)=Rnm2(mp,-m)+Rnm1(mp-1,-m)*d
               endif
               Rnm2(mp,+m)=Rnm2(mp,+m)*scale
               Rnm2(mp,-m)=Rnm2(mp,-m)*scale
               if (m.gt.mp) then
                  if(mod(mp+m,2).eq.0) then
                     Rnm2(m,+mp)=+Rnm2(mp,+m)
                     Rnm2(m,-mp)=+Rnm2(mp,-m)
                  else
                     Rnm2(m,+mp)=-Rnm2(mp,+m)
                     Rnm2(m,-mp)=-Rnm2(mp,-m)
                  endif
               endif
            enddo
         enddo
         do m=-n,n
            Mrot(n,m)=Mnm(n,0)*Rnm2(0,m)
            do mp=1,n
               Mrot(n,m)=Mrot(n,m)+
     $              Mnm(n,mp)*Rnm2(mp,m)+Mnm(n,-mp)*Rnm2(mp,-m)
            enddo
         enddo
         do m=-n,n
            do mp=0,n
               Rnm1(mp,m) = Rnm2(mp,m)
            enddo
         enddo
      enddo
      return
      end

      subroutine get_Ynm(nterms,x,Ynm,Anm1,Anm2,Pmax)
c     Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      implicit none
      integer nterms,Pmax,m,n
      real *8 x,y,Ynm(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax),Anm2(0:Pmax,0:Pmax)
      y = -sqrt((1-x)*(1+x))
      Ynm(0,0) = 1
      do m=0,nterms
         if (m.gt.0) Ynm(m,m)=Ynm(m-1,m-1)*y*Anm1(m,m)
         if (m.lt.nterms) Ynm(m+1,m)=x*Ynm(m,m)*Anm1(m+1,m)
         do n=m+2,nterms
            Ynm(n,m)=Anm1(n,m)*x*Ynm(n-1,m)
     $           -Anm2(n,m)*Ynm(n-2,m)
         enddo
      enddo
      do n=0,nterms
         do m=0,n
            Ynm(n,m)=Ynm(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine get_Ynmd(nterms,x,Ynm,Ynmd,Anm1,Anm2,Pmax)
c     Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c     d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      implicit none
      integer nterms,Pmax,m,n
      real *8 x,y,y2,Ynm(0:nterms,0:nterms),Ynmd(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax),Anm2(0:Pmax,0:Pmax)
      y=-sqrt((1-x)*(1+x))
      y2=(1-x)*(1+x)
      Ynm(0,0)=1
      Ynmd(0,0)=0
      Ynm(1,0)=x*Ynm(0,0)*Anm1(1,0)
      Ynmd(1,0)=(x*Ynmd(0,0)+Ynm(0,0))*Anm1(1,0)
      do n=2,nterms
         Ynm(n,0)=Anm1(n,0)*x*Ynm(n-1,0)-Anm2(n,0)*Ynm(n-2,0)
         Ynmd(n,0)=Anm1(n,0)*(x*Ynmd(n-1,0)
     $        +Ynm(n-1,0))-Anm2(n,0)*Ynmd(n-2,0)
      enddo
      do m=1,nterms
         if (m.eq.1) Ynm(m,m)=-Ynm(m-1,m-1)*Anm1(m,m)
         if (m.gt.1) Ynm(m,m)=Ynm(m-1,m-1)*y*Anm1(m,m)
         if (m.gt.0) Ynmd(m,m)=-Ynm(m,m)*m*x
         if (m.lt.nterms) Ynm(m+1,m)=x*Ynm(m,m)*Anm1(m+1,m)
         if (m.lt.nterms) Ynmd(m+1,m)=(x*Ynmd(m,m)+y2*Ynm(m,m))
     $        *Anm1(m+1,m)
         do n=m+2,nterms
            Ynm(n,m)=Anm1(n,m)*x*Ynm(n-1,m)
     $           -Anm2(n,m)*Ynm(n-2,m)
            Ynmd(n,m)=Anm1(n,m)*(x*Ynmd(n-1,m)+y2*Ynm(n-1,m))
     $           -Anm2(n,m)*Ynmd(n-2,m)
         enddo
      enddo
      do n=0,nterms
         do m=0,n
            Ynm(n,m)=Ynm(n,m)*sqrt(2*n+1.0d0)
            Ynmd(n,m)=Ynmd(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine get_hn(nterms,z,scale,hn)
c     hn(n) = h_n(z)*scale^(n)
      implicit none
      integer nterms,i
      real *8 scale,scale2,eps/1.0d-15/
      complex *16 z,zi,zinv,eye/(0.0d0,1.0d0)/
      complex *16 hn(0:nterms)
      if (abs(z).lt.eps) then
         do i=0,nterms
            hn(i)=0
         enddo
         return
      endif
      zi=eye*z
      zinv=scale/z
      hn(0)=exp(zi)/zi
      hn(1)=hn(0)*(zinv-eye*scale)
      scale2=scale*scale
      do i=2,nterms
         hn(i)=zinv*(2*i-1.0d0)*hn(i-1)-scale2*hn(i-2)
      enddo
      return
      end

      subroutine get_hnd(nterms,z,scale,hn,hnd)
c     hn(n) = h_n(z)*scale^(n)
c     hnd(n) = \frac{\partial hn(z)}{\partial z}
      implicit none
      integer nterms,i
      real *8 scale,eps/1.0d-15/
      complex *16 z,zi,zinv,eye/(0.0d0,1.0d0)/
      complex *16 hn(0:nterms),hnd(0:nterms)
      if (abs(z).lt.eps) then
         do i=0,nterms
            hn(i)=0
            hnd(i)=0
         enddo
         return
      endif
      zi=eye*z
      zinv=1.0/z
      hn(0)=exp(zi)/zi
      hn(1)=hn(0)*(zinv-eye)*scale
      hnd(0)=-hn(1)/scale
      hnd(1)=-zinv*2*hn(1)+scale*hn(0)
      do i=2,nterms
         hn(i)=(zinv*(2*i-1.0d0)*hn(i-1)-scale*hn(i-2))*scale
         hnd(i)=-zinv*(i+1.0d0)*hn(i)+scale*hn(i-1)
      enddo
      return
      end

      subroutine get_jn(nterms,z,scale,jn,ifder,jnd,nbessel)
c     jn(z)=j_n(z)/scale^n
c     jnd(z)=\frac{\partial jn(z)}{\partial z}
      implicit none
      integer nterms,ifder,nbessel,ntop,i,iscale(0:nbessel)
      real *8 scale,scalinv,coef,eps/1.0d-15/
      complex *16 jn(0:nbessel),jnd(0:nbessel)
      complex *16 z,zinv,fj0,fj1,ztmp
      if (abs(z).lt.eps) then
         jn(0)=1.0d0
         do i=1,nterms
            jn(i)=0.0d0
	 enddo
	 if (ifder.eq.1) then
	    do i=0,nterms
	       jnd(i)=0.0d0
	    enddo
	    jnd(1)=1.0d0/(3*scale)
	 endif
         return
      endif
      ntop=0
      zinv=1.0d0/z
      jn(nterms)=1.0d0
      jn(nterms-1)=0.0d0
      do i=nterms,nbessel
         coef=2*i+1.0d0
         ztmp=coef*zinv*jn(i)-jn(i-1)
         jn(i+1)=ztmp
         if (abs(ztmp).gt.1/eps) then
            ntop=i+1
            exit
         endif
      enddo
      if (ntop.eq.0) then
         print*,"Error: insufficient array dimension nbessel"
         stop
      endif
      do i=0,ntop
         iscale(i)=0
      enddo
      jn(ntop)=0.0d0
      jn(ntop-1)=1.0d0
      do i=ntop-1,1,-1
	 coef=2*i+1.0d0
         ztmp=coef*zinv*jn(i)-jn(i+1)
         jn(i-1)=ztmp
         if (abs(ztmp).gt.1/eps) then
            jn(i)=jn(i)*eps
            jn(i-1)=jn(i-1)*eps
            iscale(i)=1
         endif
      enddo
      scalinv=1.0d0/scale
      coef=1.0d0
      do i=1,ntop
         coef=coef*scalinv
         if(iscale(i-1).eq.1) coef=coef*eps
         jn(i)=jn(i)*coef
      enddo
      fj0=sin(z)*zinv
      fj1=fj0*zinv-cos(z)*zinv
      if (abs(fj1).gt.abs(fj0)) then
         ztmp=fj1/(jn(1)*scale)
      else
         ztmp=fj0/jn(0)
      endif
      do i=0,nterms
         jn(i)=jn(i)*ztmp
      enddo
      if (ifder.eq.1) then
         jn(nterms+1)=jn(nterms+1)*ztmp
         jnd(0)=-jn(1)*scale
         do i=1,nterms
            coef=i/(2*i+1.0d0)
            jnd(i)=coef*scalinv*jn(i-1)-(1-coef)*scale*jn(i+1)
         enddo
      endif
      return
      end

      subroutine legendre(nquad,xquad,wquad)
      implicit none
      integer nquad,i,k,ifout
      real *8 pi,h,xk,delta,pol,der,sum,eps/1.0d-15/
      real *8 xquad(10000),wquad(10000)
      pi=datan(1.0d0)*4
      h=pi/(2*nquad)
      do i=1,nquad
         xquad(nquad-i+1)=dcos((2*i-1)*h)
      enddo
      xquad(nquad/2+1)=0
      do i=1,nquad/2
         xk=xquad(i)
         ifout=0
         do k=1,10
            call polynomial(xk,nquad,pol,der,sum)
            delta=-pol/der
            xk=xk+delta
            if(abs(delta).lt.eps) ifout=ifout+1
            if(ifout.eq.3) cycle
         enddo
         xquad(i)=xk
         xquad(nquad-i+1)=-xk
      enddo
      do i=1,(nquad+1)/2
         call polynomial(xquad(i),nquad,pol,der,sum)
         wquad(i)=1/sum
         wquad(nquad-i+1)=wquad(i)
      enddo
      return
      end

      subroutine polynomial(x,n,pol,der,sum)
      implicit none
      integer n,k
      real *8 x,pol,der,sum,pk,pkp1,pkm1
      sum=0.5+x**2*1.5
      pk=1
      pkp1=x
      if(n.lt.2) then
      der=0
      sum=0.5
      if(n.eq.0) return
      der=1
      sum=sum+x**2*1.5
      return
      endif
      do k=1,n-1
         pkm1=pk
         pk=pkp1
         pkp1=((2*k+1)*x*pk-k*pkm1)/(k+1)
         sum=sum+pkp1**2*(k+1.5)
      enddo
      pol=pkp1
      der=n*(x*pkp1-pk)/(x**2-1)
      return
      end

      subroutine getNumTermsList(size,wavek,eps,itable)
      implicit none
      integer i,j,k,nterms
      integer nterms_table(2:3,0:3,0:3),itable(-3:3,-3:3,-3:3)
      real *8 size,eps,dx,dy,dz,rr
      complex *16 wavek
      do i=2,3
         do j=0,3
            do k=0,3
               dx=i
               dy=j
               dz=k
               if(dx.gt.0) dx=dx-.5
               if(dy.gt.0) dy=dy-.5
               if(dz.gt.0) dz=dz-.5
               rr=sqrt(dx*dx+dy*dy+dz*dz)
               call getNumTerms(1,rr,size,wavek,eps,nterms)
               nterms_table(i,j,k)=nterms
            enddo
         enddo
      enddo
      do i=-3,3
         do j=-3,3
            do k=-3,3
               itable(i,j,k)=0
            enddo
         enddo
      enddo
      do k=-3,3
         do i=-3,3
            do j=-3,3
               if(abs(i).gt.1) then
                  itable(i,j,k)=nterms_table(abs(i),abs(j),abs(k))
               else if(abs(j).gt.1) then
                  itable(i,j,k)=nterms_table(abs(j),abs(i),abs(k))
               endif
               if(abs(i).le.1.and.abs(j).le.1) then
                  if(abs(k).gt.1) then
                     if(abs(i).ge.abs(j)) then
                        itable(i,j,k)=nterms_table(abs(k),abs(i),abs(j))
                     else
                        itable(i,j,k)=nterms_table(abs(k),abs(j),abs(i))
                     endif
                  endif
               endif
            enddo
         enddo
      enddo
      return
      end

      subroutine getNumTerms(itype,rr,size,wavek,eps,nterms)
      implicit none
      integer itype,nterms,ntmax,j
      real *8 rr,size,eps,scale,x,x0
      complex *16 wavek,z,jn(0:2000),jnd(0:2000),hn(0:2000)
      z=(wavek*size)*rr
c     Maximum number of terms is 1000, which works for boxes up to 160 wavelengths in size
      ntmax=1000
      scale=1.0d0
      if(cdabs(wavek*size).lt.1.0d0) scale=cdabs(wavek*size)
      call get_hn(ntmax,z,scale,hn)
      z=(wavek*size)*dsqrt(3d0)/2.d0
c     corners included
      if(itype.eq.1) z=(wavek*size)*dsqrt(3d0)/2.d0
c     edges included, no corners
      if(itype.eq.2) z=(wavek*size)*dsqrt(2d0)/2.d0
c     center only
      if(itype.eq.3) z=(wavek*size)*1.0d0/2.d0
c     center only, small interior sphere
      if(itype.eq.4) z=(wavek*size)*0.8d0/2.d0
      call get_jn(ntmax,z,scale,jn,0,jnd,2000)
      x0=cdabs(jn(0)*hn(0))+cdabs(jn(1)*hn(1))
      nterms=1
      do j=2,ntmax
         x=cdabs(jn(j)*hn(j))+cdabs(jn(j-1)*hn(j-1))
         if(x.lt.eps*x0)then
            nterms=j+1
            return
         endif
      enddo
      print*,"Computational box is too big, setting nterms to 1000\n";
      nterms = 1000
      return
      end
