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

      subroutine ylgndr2(nmax, x, y, d)
      implicit none
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, du
      u=-sqrt((1-x)*(1+x))
      du=x/sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
      do m=0, nmax
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x/u**2
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
      enddo
      do n=0, nmax
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
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

      subroutine ylgndru(nmax, x, y)
      implicit none
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
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
      return
      end

      subroutine ylgndru2(nmax, x, y, d)
      implicit none
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, du
      u=-sqrt((1-x)*(1+x))
      du=x/sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
      do m=0, nmax
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x/u**2
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
      enddo
      return
      end

      subroutine ylgndru2s(nmax, x, y, d)
      implicit none
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, m, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, u2
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
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
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*sqrt(2*m+1.0d0)
	 do n=m+2, nmax
	    y(n,m)=((2*n-1)*x*y(n-1,m) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*y(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
	    d(n,m)=((2*n-1)*(x*d(n-1,m)+u2*y(n-1,m)) - 
     1               sqrt((n+m-1.0d0)*(n-m-1.0d0))*d(n-2,m))
     2               /sqrt((n-m+0.0d0)*(n+m))
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
	 if (m.gt.0)  rat2(m,m)=1
	 if (m.lt.nmax)  rat1(m+1,m)=sqrt(2*m+1.0d0)
	 if (m.lt.nmax)  rat2(m+1,m)=1
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

      subroutine ylgndru2sf(nmax, x, y, d, rat1, rat2)
      implicit none
c      Ynm(x) =  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c      d Ynm(x) / dx =  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
      integer nmax, n, m
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
      return
      end
      subroutine ylgndr2s_trunc(nmax, m2, x, y, d)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nmax
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nmax                  must be non-negative
c     x                     -1 <= x <= 1
c     y(0:nmax,0:nmax)      resulting function values
c     d(0:nmax,0:nmax)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nmax and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nmax, m2, x
cf2py intent(out) y, d
c
      integer nmax, m, m2, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
c
c       ... first, evaluate standard Legendre polynomials
c
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
c
c       ... then, evaluate scaled associated Legendre functions
c
      do m=1, m2
c
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*sqrt((2*m-1.0d0)/(2*m))
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
c
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
c
      do n=0, nmax
	 do m=0, min(n,m2)
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end
c
c
c
c
c
c
      subroutine ylgndrf_trunc(nmax, m2, x, y, rat1, rat2)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nmax
c     and  m = 0, 1,..., min(m2,n).
c
c     Parameters:
c     nmax                  must be non-negative
c     x                     -1 <= x <= 1
c     y(0:nmax,0:nmax)      resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nmax and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nmax, m2, x, rat1, rat2
cf2py intent(out) y
c
      integer nmax, m, m2, n
      real *8 x, y(0:nmax,0:nmax), u
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      do m=0, m2
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 do n=m+2, nmax
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         enddo
      enddo
c
      do n=0, nmax
	 do m=0, min(n,m2)
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2f_trunc(nmax, m2, x, y, d, rat1, rat2)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c    
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c    
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c    
c     for n = 0, 1, 2,..., nmax
c     and  m = 0, 1,..., min(m2,n).
c    
c     Parameters:
c     nmax                  must be non-negative
c     x                     -1 <= x <= 1
c     y(0:nmax,0:nmax)      resulting function values
c     d(0:nmax,0:nmax)      resulting derivative values
c    
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nmax and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nmax, m2, x, rat1, rat2
cf2py intent(out) y, d
c
      integer nmax, m, m2, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, du
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      du=x/sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
      do m=0, m2
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x/u**2
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nmax)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
	 do n=m+2, nmax
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
         enddo
      enddo
c
      do n=0, nmax
	 do m=0, min(n,m2)
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2sf_trunc(nmax, m2, x, y, d, rat1, rat2)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nmax
c     and  m = 0, 1,..., min(n,m2).
c
c     Parameters:
c     nmax                  must be non-negative
c     x                     -1 <= x <= 1
c     y(0:nmax,0:nmax)      resulting function values
c     d(0:nmax,0:nmax)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nmax and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nmax, m2, x, rat1, rat2
cf2py intent(out) y, d
c
      integer nmax, m, m2, n
      real *8 x, y(0:nmax,0:nmax), d(0:nmax,0:nmax), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
c
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
      y(0,0)=1
      d(0,0)=0
c
c       ... first, evaluate standard Legendre polynomials
c
      m=0
      if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      if (m.lt.nmax)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
      do n=m+2, nmax
        y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
        d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
      enddo
c
c       ... then, evaluate scaled associated Legendre functions
c
      do m=1, m2
c
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
c
	 if (m.lt.nmax)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nmax)  
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
	 do n=m+2, nmax
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+u2*y(n-1,m))-
     $         rat2(n,m)*d(n-2,m)
         enddo
      enddo
c
      do n=0, nmax
	 do m=0, min(n,m2)
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
      return
      end
c
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c       Symmetries for Legendre functions
c    
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
        subroutine ylgndrpm(nterms,y)
        implicit none
        integer n,m,nterms
        real *8 y(0:nterms,0:nterms)
c
c       Given Y_nm(x), return Y_nm(-x)
c
        do n=0,nterms
           do m=0,n
              if( mod(n+m,2) .eq. 1 ) y(n,m)=-y(n,m)
           enddo       
        enddo
c
        return
        end
c
c
c
c
c
        subroutine ylgndr2pm(nterms,y,d)
        implicit none
        integer nterms,n,m
        real *8 y(0:nterms,0:nterms)
        real *8 d(0:nterms,0:nterms)
c
c       Given Y_nm(x), return Y_nm(-x)
c       Given Y'_nm(x), return Y'_nm(-x)
c
cf2py intent(in) nterms, y
cf2py intent(out) d
c
        do n=0,nterms
           do m=0,n
              if( mod(n+m,2) .eq. 1 ) y(n,m)=-y(n,m)
              if( mod(n+m,2) .eq. 0 ) d(n,m)=-d(n,m)
           enddo       
        enddo
c
        return
        end
c
c
c
c
c
        subroutine ylgndrpm_opt(nterms,y)
        implicit none
        integer nterms,n,m
        real *8 y(0:nterms,0:nterms)
c
c       Given Y_nm(x), return Y_nm(-x)
c
        do n=0,nterms,2
           do m=1,n,2
              y(n,m)=-y(n,m)
           enddo       
        enddo
c
        do n=1,nterms,2
           do m=0,n,2
              y(n,m)=-y(n,m)
           enddo       
        enddo
c
        return
        end
c
c
c
c
c
        subroutine ylgndr2pm_opt(nterms,y,d)
        implicit none
        integer nterms,n,m
        real *8 y(0:nterms,0:nterms)
        real *8 d(0:nterms,0:nterms)
c
c       Given Y_nm(x), return Y_nm(-x)
c       Given Y'_nm(x), return Y'_nm(-x)
c
cf2py intent(in) nterms, y
cf2py intent(out) d
c
        do n=0,nterms,2
           do m=0,n,2
              d(n,m)=-d(n,m)
           enddo       
           do m=1,n,2
              y(n,m)=-y(n,m)
           enddo       
        enddo
c
        do n=1,nterms,2
           do m=0,n,2
              y(n,m)=-y(n,m)
           enddo       
           do m=1,n,2
              d(n,m)=-d(n,m)
           enddo       
        enddo
c
        return
        end
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c       fast version of real valued Legendre functions, with storage
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
      subroutine ylgndrfwini(nmax, w, lw, lused)
      implicit none
c
c     Precompute the recurrence coefficients for the fast
c     evaluation of normalized Legendre functions and their derivatives
c    
c     Parameters:
c       nmax             must be non-negative
c       w                  contains rat1 and rat2 arrays
c
      integer nmax,irat1,irat2,lw,lused
      real *8 w(*)
c
cf2py intent(in) nmax, lw
cf2py intent(out) w, lused
c
      irat1=1
      irat2=1+(nmax+1)**2
      lused=2*(nmax+1)**2
      if( lused .gt. lw ) return
      
      call ylgndrini(nmax, w(irat1), w(irat2))
      return
      end
c
c
c
c
c
      subroutine ylgndrfw(nterms, x, y, w, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)          resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndrfw0(nterms, x, y, w(irat1), w(irat2), nmax)
      else
        call ylgndr(nterms, x, y)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2fw(nterms, x, y, d, w, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y, d
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndr2fw0(nterms, x, y, d, w(irat1), w(irat2), nmax)
      else
        call ylgndr2(nterms, x, y, d)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2sfw(nterms, x, y, d, w, nmax)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y, d
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndr2sfw0(nterms, x, y, d, w(irat1), w(irat2), nmax)
      else
        call ylgndr2s(nterms, x, y, d)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndrfw0(nterms, x, y, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y
c
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
c
      do n=0, nterms
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2fw0(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, m, n
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, du
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      du=x/sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
      do m=0, nterms
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x/u**2
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nterms)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
	 do n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
         enddo
      enddo
c
      do n=0, nterms
	 do m=0, n
	    y(n,m)=y(n,m)*sqrt(2*n+1.0d0)
	    d(n,m)=d(n,m)*sqrt(2*n+1.0d0)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndr2sfw0(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt(2n+1)  sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, m, n
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
      y(0,0)=1
      d(0,0)=0
c
c       ... first, evaluate standard Legendre polynomials
c
      m=0
      if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      if (m.lt.nterms)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
      do n=m+2, nterms
        y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
        d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
      enddo
c
c       ... then, evaluate scaled associated Legendre functions
c
      do m=1, nterms
c
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
c
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nterms)  
ccc     $      d(m+1,m)=(x*d(m,m)+(1-x**2)*y(m,m))*rat1(m+1,m)
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
	 do n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
ccc	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+(1-x**2)*y(n-1,m))-
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
c
c
c
c
c
      subroutine ylgndrufw(nterms, x, y, w, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)          resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndrufw0(nterms, x, y, w(irat1), w(irat2), nmax)
      else
        call ylgndru(nterms, x, y)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndru2fw(nterms, x, y, d, w, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y, d
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndru2fw0(nterms, x, y, d, w(irat1), w(irat2), nmax)
      else
        call ylgndru2(nterms, x, y, d)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndru2sfw(nterms, x, y, d, w, nmax)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, w, nmax
cf2py intent(out) y, d
c
      integer nterms,nmax,irat1,irat2
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), w(*)
c
      if( nterms .le. nmax ) then
        irat1=1
        irat2=1+(nmax+1)**2
        call ylgndru2sfw0(nterms, x, y, d, w(irat1), w(irat2), nmax)
      else
        call ylgndru2s(nterms, x, y, d)
      endif
c
      return
      end
c
c
c
c
c
      subroutine ylgndrufw0_old(nterms, x, y, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), u
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      y(0,0)=1
      do 10 m=0, nterms
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 do 20 n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
 20      continue
 10   continue
c
      return
      end
c
c
c
c
c
      subroutine ylgndrufw0(nterms, x, y, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. 
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), u
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)

      y(0,0)=1
      if( nterms .eq. 0 ) return

      y(1,0)=x*y(0,0)*rat1(1,0)

      u=-sqrt((1-x)*(1+x))
      do m=1, nterms-1
	 y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      enddo

      y(nterms,nterms)=y(nterms-1,nterms-1)*u*rat1(nterms,nterms)

c      do m=0, nterms-1
c	 do n=m+2, nterms
c	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
c         enddo
c      enddo

      do n=2, nterms
         do m=0, n-2
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndru2fw0_old(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, du
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      du=x/sqrt((1-x)*(1+x))
      y(0,0)=1
      d(0,0)=0
      do 10 m=0, nterms
	 if (m.gt.0)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x/u**2
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nterms)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
	 do 20 n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
 20      continue
 10   continue
c
      return
      end
c
c
c
c
c
      subroutine ylgndru2fw0(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate normalized Legendre functions and their derivatives
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)

      y(0,0)=1
      d(0,0)=0
      if( nterms .eq. 0 ) return

      y(1,0)=x*y(0,0)*rat1(1,0)
      d(1,0)=(x*d(0,0)+y(0,0))*rat1(1,0)

      u=-sqrt((1-x)*(1+x))
      u2=(1-x)*(1+x)
      do m=1, nterms-1
         y(m,m)=y(m-1,m-1)*u*rat1(m,m)
         d(m,m)=y(m,m)*(-m)*x/u2
         y(m+1,m)=x*y(m,m)*rat1(m+1,m)
         d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
      enddo

      y(nterms,nterms)=y(nterms-1,nterms-1)*u*rat1(nterms,nterms)
      d(nterms,nterms)=y(nterms,nterms)*(-nterms*x)

      do n=2, nterms
         do m=0, n-2
         y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
         enddo
      enddo
c
      return
      end
c
c
c
c
c
      subroutine ylgndru2sfw0_old(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)
      u=-sqrt((1-x)*(1+x))
      u2 = (1-x)*(1+x)
      y(0,0)=1
      d(0,0)=0
c
c       ... first, evaluate standard Legendre polynomials
c
      m=0
      if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
      if (m.lt.nterms)  d(m+1,m)=(x*d(m,m)+y(m,m))*rat1(m+1,m)
      do 120 n=m+2, nterms
        y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
        d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
120   continue
c
c       ... then, evaluate scaled associated Legendre functions
c
      do 210 m=1, nterms
c
	 if (m.eq.1)  y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
	 if (m.gt.1)  y(m,m)=y(m-1,m-1)*u*rat1(m,m)
	 if (m.gt.0)  d(m,m)=y(m,m)*(-m)*x
c
	 if (m.lt.nterms)  y(m+1,m)=x*y(m,m)*rat1(m+1,m)
	 if (m.lt.nterms)  
     $      d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
	 do 220 n=m+2, nterms
	    y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
	    d(n,m)=rat1(n,m)*(x*d(n-1,m)+u2*y(n-1,m))-
     $         rat2(n,m)*d(n-2,m)
220      continue
210   continue
c
      return
      end
c
c
c

      subroutine ylgndru2sfw0(nterms, x, y, d, rat1, rat2, nmax)
      implicit none
c
c     Evaluate scaled normalized Legendre functions and their derivatives
c
c     Only for Ynm(x) with m>0 
c          the functions are scaled by 1/sqrt(1-x**2)
c          the derivatives are scaled by sqrt(1-x**2)
c
c
c      Ynm(x) = sqrt( (n-m)!/ (n+m)! ) Pnm(x)
c
c      d Ynm(x) / dx = sqrt( (n-m)!/ (n+m)! ) d Pnm(x) / dx
c
c     for n = 0, 1, 2,..., nterms
c     and  m = 0, 1,..., n.
c
c     Parameters:
c     nterms                      must be non-negative
c     x                         -1 <= x <= 1
c     y(0:nterms,0:nterms)      resulting function values
c     d(0:nterms,0:nterms)      resulting derivative values
c
c     Upon return, y(n,m) will contain the function value Ynm(x) for 0
c     <= n <= nterms and 0 <= m <= n.  Other elements of y will contain
c     undefined values. The same convention for the derivatives.
c
cf2py intent(in) nterms, x, rat1, rat2, nmax
cf2py intent(out) y, d
c
      integer nterms, nmax, n, m
      real *8 x, y(0:nterms,0:nterms), d(0:nterms,0:nterms), u, u2
      real *8 rat1(0:nmax,0:nmax), rat2(0:nmax,0:nmax)

      y(0,0)=1
      d(0,0)=0
      if( nterms .eq. 0 ) return

      y(1,0)=x*y(0,0)*rat1(1,0)
      d(1,0)=(x*d(0,0)+y(0,0))*rat1(1,0)

      u=-sqrt((1-x)*(1+x))
      u2=(1-x)*(1+x)
      do m=1, nterms-1
         if( m .eq. 1 ) y(m,m)=y(m-1,m-1)*(-1)*rat1(m,m)
         if( m .gt. 1 ) y(m,m)=y(m-1,m-1)*u*rat1(m,m)
         d(m,m)=y(m,m)*(-m)*x
         y(m+1,m)=x*y(m,m)*rat1(m+1,m)
         d(m+1,m)=(x*d(m,m)+u2*y(m,m))*rat1(m+1,m)
      enddo

      y(nterms,nterms)=y(nterms-1,nterms-1)*u*rat1(nterms,nterms)
      d(nterms,nterms)=y(nterms,nterms)*(-nterms*x)

      do n=2, nterms
         m=0
         y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         d(n,m)=rat1(n,m)*(x*d(n-1,m)+y(n-1,m))-rat2(n,m)*d(n-2,m)
         do m=1, n-2
         y(n,m)=rat1(n,m)*x*y(n-1,m)-rat2(n,m)*y(n-2,m)
         d(n,m)=rat1(n,m)*(x*d(n-1,m)+u2*y(n-1,m))-rat2(n,m)*d(n-2,m)
         enddo
      enddo
c
      return
      end
