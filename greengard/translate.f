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

      subroutine initCoefs(C, nterms)
      implicit none
      integer nterms, m, n
      complex *16 C(0:nterms,-nterms:nterms)
      do n=0,nterms
         do m=-n,n
            C(n,m)=0
         enddo
      enddo
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
c**********************************************************************
      subroutine bessel(nterms,z,scale,fjs,ifder,fjder,nbessel)
      implicit none
      integer nterms,ifder,nbessel,ntop,i,ncntr
      real *8 scale,d0,d1,dc1,dc2,dcoef,dd,done,tiny,zero
      real *8 scalinv,sctot,upbound,upbound2,upbound2inv
c**********************************************************************
c       This subroutine evaluates the first NTERMS spherical Bessel
c       functions and if required, their derivatives.
c       It incorporates a scaling parameter SCALE so that
c
c               fjs_n(z)=j_n(z)/SCALE^n
c               fjder_n(z)=\frac{\partial fjs_n(z)}{\partial z}
c---------------------------------------------------------------------
c     INPUT:
c     nterms  : order of expansion of output array fjs
c     z       : argument of the spherical Bessel functions
c     scale   : scaling factor (discussed above)
c     ifder   : flag indicating whether to calculate "fjder"
c     nbessel : upper limit of input arrays
c     fjs(0:nbessel) and iscale(0:nbessel)
c     iscale  : integer workspace used to keep track of
c     internal scaling
c---------------------------------------------------------------------
c     OUTPUT:
c     fjs     : array of scaled Bessel functions.
c     fjder   : array of derivs of scaled Bessel functions.
c     ntop    : highest index in arrays fjs that is nonzero
c     NOTE, that fjs and fjder arrays must be at least (nterms+2)
c     complex *16 elements long.
c---------------------------------------------------------------------
      integer iscale(0:nbessel)
      complex *16 wavek,fjs(0:nbessel),fjder(0:*)
      complex *16 z,zinv,com,fj0,fj1,zscale,ztmp
      data upbound/1.0d+32/, upbound2/1.0d+40/, upbound2inv/1.0d-40/
      data tiny/1.0d-200/,done/1.0d0/,zero/0.0d0/
c       set to asymptotic values if argument is sufficiently small
      if (abs(z).lt.tiny) then
         fjs(0) = done
         do i = 1, nterms
            fjs(i) = zero
	 enddo
	 if (ifder.eq.1) then
	    do i=0,nterms
	       fjder(i)=zero
	    enddo
	    fjder(1)=done/(3*scale)
	 endif
         return
      endif
c ... Step 1: recursion up to find ntop, starting from nterms
      ntop=0
      zinv=done/z
      fjs(nterms)=done
      fjs(nterms-1)=zero
      do i=nterms,nbessel
         dcoef=2*i+done
         ztmp=dcoef*zinv*fjs(i)-fjs(i-1)
         fjs(i+1)=ztmp
         dd = dreal(ztmp)**2 + dimag(ztmp)**2
         if (dd .gt. upbound2) then
            ntop=i+1
            exit
         endif
      enddo
      if (ntop.eq.0) then
         print*,"Error: insufficient array dimension nbessel"
         stop
      endif
c ... Step 2: Recursion back down to generate the unscaled jfuns:
c             if magnitude exceeds UPBOUND2, rescale and continue the
c	      recursion (saving the order at which rescaling occurred
c	      in array iscale.
      do i=0,ntop
         iscale(i)=0
      enddo
      fjs(ntop)=zero
      fjs(ntop-1)=done
      do i=ntop-1,1,-1
	 dcoef=2*i+done
         ztmp=dcoef*zinv*fjs(i)-fjs(i+1)
         fjs(i-1)=ztmp
         dd = dreal(ztmp)**2 + dimag(ztmp)**2
         if (dd.gt.UPBOUND2) then
            fjs(i) = fjs(i)*UPBOUND2inv
            fjs(i-1) = fjs(i-1)*UPBOUND2inv
            iscale(i) = 1
         endif
      enddo
c ...  Step 3: go back up to the top and make sure that all
c              Bessel functions are scaled by the same factor
c              (i.e. the net total of times rescaling was invoked
c              on the way down in the previous loop).
c              At the same time, add scaling to fjs array.
      ncntr=0
      scalinv=done/scale
      sctot = 1.0d0
      do i=1,ntop
         sctot = sctot*scalinv
         if(iscale(i-1).eq.1) sctot=sctot*UPBOUND2inv
         fjs(i)=fjs(i)*sctot
      enddo
c ... Determine the normalization parameter:
      fj0=sin(z)*zinv
      fj1=fj0*zinv-cos(z)*zinv
      d0=abs(fj0)
      d1=abs(fj1)
      if (d1 .gt. d0) then
         zscale=fj1/(fjs(1)*scale)
      else
         zscale=fj0/fjs(0)
      endif
c ... Scale the jfuns by zscale:
      ztmp=zscale
      do i=0,nterms
         fjs(i)=fjs(i)*ztmp
      enddo
c ... Finally, calculate the derivatives if desired:
      if (ifder.eq.1) then
         fjs(nterms+1)=fjs(nterms+1)*ztmp
         fjder(0)=-fjs(1)*scale
         do i=1,nterms
            dc1=i/(2*i+done)
            dc2=done-dc1
            dc1=dc1*scalinv
            dc2=dc2*scale
            fjder(i)=dc1*fjs(i-1)-dc2*fjs(i+1)
         enddo
      endif
      return
      end
c**********************************************************************
      subroutine legendre(n,ts,whts,ifwhts)
c**********************************************************************
c     This subroutine constructs the nodes and the
c     weights of the n-point gaussian quadrature on 
c     the interval [-1,1]
c---------------------------------------------------------------------
c     INPUT:
c     n  : the number of nodes in the quadrature
c---------------------------------------------------------------------
c     OUTPUT:
c     ts : the nodes of the n-point gaussian quadrature
c     w  : the weights of the n-point gaussian quadrature
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      dimension ts(1),whts(1),ws2(1000),rats(1000)
      data eps/1.0d-14/
      ZERO=0
      DONE=1
      pi=datan(done)*4
      h=pi/(2*n) 
      do i=1,n
         t=(2*i-1)*h
         ts(n-i+1)=dcos(t)
      enddo
c     use newton to find all roots of the legendre polynomial
      ts(n/2+1)=0
      do i=1,n/2
         xk=ts(i)
         ifout=0
         deltold=1
         do k=1,10
            call polynomial(xk,n,pol,der,sum)
            delta=-pol/der
            xk=xk+delta
            if(abs(delta) .lt. eps) ifout=ifout+1
            if(ifout .eq. 3) cycle
         enddo
         ts(i)=xk
         ts(n-i+1)=-xk
      enddo
c     construct the weights via the orthogonality relation
      if(ifwhts .eq. 0) return
      do i=1,(n+1)/2
         call polynomial(ts(i),n,pol,der,sum)
         whts(i)=1/sum
         whts(n-i+1)=whts(i)
      enddo
      return
      end

      subroutine polynomial(x,n,pol,der,sum)
      implicit real *8 (a-h,o-z)
      sum=0 
      pkm1=1
      pk=x
      sum=sum+pkm1**2 /2
      sum=sum+pk**2 * 1.5
      pk=1
      pkp1=x
      if(n .lt. 2) then
      sum=0 
      pol=1
      der=0
      sum=sum+pol**2 /2
      if(n .eq. 0) return
      pol=x
      der=1
      sum=sum+pol**2*1.5
      return
      endif
c     n is greater than 1. conduct recursion
      do k=1,n-1
         pkm1=pk
         pk=pkp1
         pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
         sum=sum+pkp1**2*(k+1.5)
      enddo
c     calculate the derivative
      pol=pkp1
      der=n*(x*pkp1-pk)/(x**2-1)
      return
      end
