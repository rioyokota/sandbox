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

      subroutine getAnm(Pmax, Anm1, Anm2)
      implicit none
      integer Pmax, m, n
      real *8 Anm1(0:Pmax,0:Pmax), Anm2(0:Pmax,0:Pmax)
      Anm1(0,0) = 1
      Anm2(0,0) = 1
      do m=0, Pmax
         if (m.gt.0) Anm1(m,m) = sqrt((2 * m - 1.0d0) / (2 * m))
         if (m.lt.Pmax) Anm1(m+1,m) = sqrt(2 * m + 1.0d0)
         do n=m+2, Pmax
            Anm1(n,m) = (2*n-1)
            Anm2(n,m) = sqrt((n + m - 1.0d0) * (n - m - 1.0d0))
            Anm1(n,m) = Anm1(n,m) / sqrt(dble(n - m) * (n + m))
            Anm2(n,m) = Anm2(n,m) / sqrt(dble(n - m) * (n + m))
         enddo
      enddo
      return
      end

      subroutine getYnm(nterms, x, Ynm, Anm1, Anm2, Pmax)
c     Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      implicit none
      integer nterms, Pmax, m, n
      real *8 x, y, Ynm(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax), Anm2(0:Pmax,0:Pmax)
      y = -sqrt((1 - x) * (1 + x))
      Ynm(0,0) = 1
      do m=0, nterms
         if (m.gt.0) Ynm(m,m) = Ynm(m-1,m-1) * y * Anm1(m,m)
         if (m.lt.nterms) Ynm(m+1,m) = x * Ynm(m,m) * Anm1(m+1,m)
         do n=m+2, nterms
            Ynm(n,m) = Anm1(n,m) * x * Ynm(n-1,m)
     $           - Anm2(n,m) * Ynm(n-2,m)
         enddo
      enddo
      do n=0, nterms
         do m=0, n
            Ynm(n,m) = Ynm(n,m) * sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine getYnmd(nterms, x, Ynm, Ynmd, Anm1, Anm2, Pmax)
c     Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c     d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      implicit none
      integer nterms, Pmax, m, n
      real *8 x, y, y2, Ynm(0:nterms,0:nterms), Ynmd(0:nterms,0:nterms)
      real *8 Anm1(0:Pmax,0:Pmax), Anm2(0:Pmax,0:Pmax)
      y = -sqrt((1 - x) * (1 + x))
      y2 = (1 - x) * (1 + x)
      Ynm(0,0) = 1
      Ynmd(0,0) = 0
      Ynm(1,0) = x * Ynm(0,0) * Anm1(1,0)
      Ynmd(1,0) = (x * Ynmd(0,0) + Ynm(0,0)) * Anm1(1,0)
      do n=2, nterms
         Ynm(n,0) = Anm1(n,0) * x * Ynm(n-1,0) - Anm2(n,0) * Ynm(n-2,0)
         Ynmd(n,0) = Anm1(n,0) * (x * Ynmd(n-1,0)
     $        + Ynm(n-1,0)) - Anm2(n,0) * Ynmd(n-2,0)
      enddo
      do m=1, nterms
         if (m.eq.1) Ynm(m,m) = -Ynm(m-1,m-1) * Anm1(m,m)
         if (m.gt.1) Ynm(m,m) = Ynm(m-1,m-1) * y * Anm1(m,m)
         if (m.gt.0) Ynmd(m,m) = -Ynm(m,m) * m * x
         if (m.lt.nterms) Ynm(m+1,m) = x * Ynm(m,m) * Anm1(m+1,m)
         if (m.lt.nterms) Ynmd(m+1,m) = (x * Ynmd(m,m)
     $        + y2 * Ynm(m,m)) * Anm1(m+1,m)
         do n=m+2, nterms
            Ynm(n,m) = Anm1(n,m) * x * Ynm(n-1,m) 
     $           - Anm2(n,m) * Ynm(n-2,m)
            Ynmd(n,m) = Anm1(n,m) * (x * Ynmd(n-1,m) + y2 * Ynm(n-1,m))
     $           - Anm2(n,m) * Ynmd(n-2,m)
         enddo
      enddo
      do n=0, nterms
         do m=0, n
            Ynm(n,m) = Ynm(n,m) * sqrt(2*n+1.0d0)
            Ynmd(n,m) = Ynmd(n,m) * sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

c*****************************************************************
      subroutine rotate(theta,ntermsj,Mnm,ntermsi,Mrot)
c*****************************************************************
c     Fast, recursive method for applying rotation matrix about
c     the y-axis determined by angle theta.
c     The rotation matrices for each order (first index) are computed
c     from the lowest to the highest. As each one is generated, it
c     is applied to the input expansion "Mnm" and overwritten.
c     
c     As a result, it is sufficient to use two arrays rd1 and rd2 for
c     the two term recurrence, rather than storing them for all orders
c     as in the original code. There is some loss in speed
c     if the rotation operator is to be used multiple times, but the
c     memory savings is often more critical.
c     
c     Use symmetry properties of rotation matrices
c---------------------------------------------------------------------
c     INPUT:
c     ntermsj : dimension parameter for d - the rotation matrix.
c     Mnm     : coefficients of original multiple expansion
c     rd1     : work space
c     rd2     : work space
c     sqc:    : square roots of the binomial coefficients.
c     theta   : the rotate angle about the y-axis.
c---------------------------------------------------------------------
c     OUTPUT:
c     Mrot    : coefficients of rotated expansion.
c---------------------------------------------------------------------
      implicit none
      integer ntermsi,ntermsj
      integer ij,im,imp,m,mp
      real *8 theta
      real *8 ww,done,ctheta,stheta,hsthta,cthtap,cthtan,d
      real *8 precis,scale
      real *8 rd1(0:ntermsj,-ntermsj:ntermsj)
      real *8 rd2(0:ntermsj,-ntermsj:ntermsj)
      real *8 sqc(0:2*ntermsj,2)
      complex *16 Mnm(0:ntermsj,-ntermsj:ntermsj)
      complex *16 Mrot(0:ntermsi,-ntermsi:ntermsi)
      data precis/1.0d-20/
      ww=1/sqrt(2.0d0)
      do m = 0, 2*ntermsj
         sqc(m,1) = dsqrt(m+0.0d0)
      enddo
      sqc(0,2) = 0.0d0
      if( ntermsj .gt. 0 ) sqc(1,2) = 0.0d0
      do m = 2, 2*ntermsj
         sqc(m,2) = dsqrt((m+0.0d0)*(m-1)/2.0d0)
      enddo
      done=1
      ctheta=dcos(theta)
      if (dabs(ctheta).le.precis) ctheta=0.0d0
      stheta=dsin(-theta)
      if (dabs(stheta).le.precis) stheta=0.0d0
      hsthta=ww*stheta
      cthtap=+2.0d0*ww*dcos(theta/2.0d0)**2
      cthtan=-2.0d0*ww*dsin(theta/2.0d0)**2
c     Compute the (0,0,0) term.
      rd1(0,0)=done
      Mrot(0,0)=Mnm(0,0)*rd1(0,0)
c     Loop over first index ij=1,ntermsj, constructing
c     rotation matrices recursively.
      do ij=1,ntermsj
c     For mprime=0, use formula (1).
         do im=-ij,-1
            rd2(0,im)=-sqc(ij-im,2)*rd1(0,im+1)
            if (im.gt.(1-ij)) then
               rd2(0,im)=rd2(0,im)+sqc(ij+im,2)*rd1(0,im-1)
            endif
            rd2(0,im)=rd2(0,im)*hsthta
            if (im.gt.-ij) then
               rd2(0,im)=rd2(0,im)+
     1              rd1(0,im)*ctheta*sqc(ij+im,1)*sqc(ij-im,1)
            endif
            rd2(0,im)=rd2(0,im)/ij
         enddo
         rd2(0,0)=rd1(0,0)*ctheta
         if (ij.gt.1) then
            rd2(0,0)=rd2(0,0)+hsthta*sqc(ij,2)*(2*rd1(0,-1))/ij
         endif
         do im=1,ij
            rd2(0,im)=rd2(0,-im)
            if( mod(im,2) .eq. 0 ) then
               rd2(im,0)=+rd2(0,im)
            else
               rd2(im,0)=-rd2(0,im)
            endif
         enddo
c     For 0<mprime<=j (2nd index) case, use formula (2).
         do imp=1,ij
            scale=(ww/sqc(ij+imp,2))
            do im=imp,ij
               rd2(imp,+im)=rd1(imp-1,+im-1)*(cthtap*sqc(ij+im,2))
               rd2(imp,-im)=rd1(imp-1,-im+1)*(cthtan*sqc(ij+im,2))
               if (im.lt.(ij-1)) then
                  rd2(imp,+im)=rd2(imp,+im)-rd1(imp-1,+im+1)*
     $                 (cthtan*sqc(ij-im,2))
                  rd2(imp,-im)=rd2(imp,-im)-rd1(imp-1,-im-1)*
     $                 (cthtap*sqc(ij-im,2))
               endif
               if (im.lt.ij) then
                  d=(stheta*sqc(ij+im,1)*sqc(ij-im,1))
                  rd2(imp,+im)=rd2(imp,+im)+rd1(imp-1,+im)*d
                  rd2(imp,-im)=rd2(imp,-im)+rd1(imp-1,-im)*d
               endif
               rd2(imp,+im)=rd2(imp,+im)*scale
               rd2(imp,-im)=rd2(imp,-im)*scale
               if (im.gt.imp) then
                  if( mod(imp+im,2) .eq. 0 ) then
                     rd2(im,+imp)=+rd2(imp,+im)
                     rd2(im,-imp)=+rd2(imp,-im)
                  else
                     rd2(im,+imp)=-rd2(imp,+im)
                     rd2(im,-imp)=-rd2(imp,-im)
                  endif
               endif
            enddo
         enddo
         do m=-ij,ij
            Mrot(ij,m)=Mnm(ij,0)*rd2(0,m)
            do mp=1,ij
               Mrot(ij,m)=Mrot(ij,m)+
     1              Mnm(ij,mp)*rd2(mp,m)+
     1              Mnm(ij,-mp)*rd2(mp,-m)
            enddo
         enddo
         do m=-ij,ij
            do mp=0,ij
               rd1(mp,m) = rd2(mp,m)
            enddo
         enddo
      enddo
      return
      end
