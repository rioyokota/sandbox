      subroutine ylgndr(nmax, x, y)
      implicit none
c     Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), u
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      do m=0, nmax
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*sqrt((2*m-1.0d0)/(2*m))
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*sqrt(2*m+1.0d0)
	 do n=m+2, nmax
	    y(n,m)=((2*n-1)*x*y(n-1,m) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*y(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
         enddo
      enddo
      do n=0, nmax
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine ylgndr2s(nmax, x, y, d)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
c       ... first, evaluate standard Legendre polynomials
      m=0
      if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*sqrt(2*m+1.0d0)
      if (m.lt.nmax)  d(m+1,m)=(x*d(m,m)+y(m,m))*sqrt(2*m+1.0d0)
      do n=m+2, nmax
        y(n,m)=((2*n-1)*x*y(n-1,m) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*y(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
        d(n,m)=((2*n-1)*(x*d(n-1,m)+y(n-1,m)) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*d(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
      enddo
c       ... then, evaluate scaled associated Legendre functions
      do m=1, nmax
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*sqrt(2*m+1.0d0)
	 if (m.lt.nmax)  
     $      d(m+1,m)=(x*d(m,m)+(1-x**2)*y(m,m))*sqrt(2*m+1.0d0)
	 do n=m+2, nmax
	    y(n,m)=((2*n-1)*x*y(n-1,m) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*y(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
	    d(n,m)=((2*n-1)*(x*d(n-1,m)+(1-x**2)*y(n-1,m)) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*d(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
         enddo
      enddo
      do n=0, nmax
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine ylgndrini(nmax, rat1, rat2)
      implicit none
c     Precompute the recurrence coefficients for the fast
c     evaluation of normalized Legendre functions and their derivatives
      integer nmax, m, n
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      rat1(0,0)=1
      rat2(0,0)=1
      do m=0, nmax
	 if (m.gt.0)  rat1(m,m)=sqrt((2*m-1.0d0)/(2*m))
	 if (m.lt.nmax)  rat1(m+1,m)=sqrt(2*m+1.0d0)
	 do n=m+2, nmax
	    rat1(n,m)=(2*n-1)
            rat2(n,m)=sqrt((n+m-1.0d0)*(n-m-1.0d0))
	    rat1(n,m)=rat1(n,m)/sqrt(dble(n-m)*(n+m))
	    rat2(n,m)=rat2(n,m)/sqrt(dble(n-m)*(n+m))
         enddo
      enddo
      return
      end

      subroutine ylgndrf(nmax, x, y, rat1, rat2)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), u
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      do m=0, nmax
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 do n=m+2, nmax
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         enddo
      enddo
      do n=0, nmax
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine ylgndr2sf(nmax, x, y, d, rat1, rat2)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
      y(0,0)=1
      d(0,0)=0
c       ... first, evaluate standard Legendre polynomials
      m=0
      if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      if (m.lt.nmax)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
      do n=m+2, nmax
        y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
        d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
      enddo
c       ... then, evaluate scaled associated Legendre functions
      do m=1, nmax
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nmax)  
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
	 do n=m+2, nmax
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+u2*y(n-1,m))-
     $         rat2(n,m)*d(n-2,m)
         enddo
      enddo
      do n=0, nmax
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine ylgndrfwini(nmax, w, lw, lused)
      implicit none
c     Precompute the recurrence coefficients for the fast
c     evaluation of normalized Legendre functions and their derivatives
      integer nmax,irat1,irat2,lw,lused
      real *8 w(*)
      irat1=1
      irat2=1+(nmax+1)**2
      lused=2*(nmax+1)**2
      if( lused .gt. lw ) return
      call ylgndrini(nmax, w(irat1), w(irat2))
      return
      end

      subroutine ylgndrfw(nterms, x, y, w, nmax)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), w(*)
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndrfw0(nterms, x, y, w(irat1), w(irat2), nmax)
      else
        call ylgndr(nterms, x, y)
      endif
      return
      end

      subroutine ylgndr2sfw(nterms, x, y, d, w, nmax)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), w(*)
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndr2sfw0(nterms, x, y, d, w(irat1), w(irat2), nmax)
      else
        call ylgndr2s(nterms, x, y, d)
      endif
      return
      end

      subroutine ylgndrfw0(nterms, x, y, rat1, rat2, nmax)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
      integer nterms,nmax,m,n
      real *8 x, y(0:nterms,0:nterms), u
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      do m=0, nterms
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 do n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         enddo
      enddo
      do n=0, nterms
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end

      subroutine ylgndr2sfw0(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nterms, nmax, m, n
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
      y(0,0)=1
      d(0,0)=0
c       ... first, evaluate standard Legendre polynomials
      m=0
      if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      if (m.lt.nterms)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
      do n=m+2, nterms
        y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
        d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
      enddo
c       ... then, evaluate scaled associated Legendre functions
      do m=1, nterms
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nterms)  
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
	 do n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+u2*y(n-1,m))-
     $         rat2(n,m)*d(n-2,m)
         enddo
      enddo
      do n=0, nterms
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end
