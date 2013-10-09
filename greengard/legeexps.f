cc Copyright (C) 2009: Vladimir Rokhlin
cc 
cc This program is free software; you can redistribute it and/or modify 
cc it under the terms of the GNU General Public License as published by 
cc the Free Software Foundation; either version 2 of the License, or 
cc (at your option) any later version.  This program is distributed in 
cc the hope that it will be useful, but WITHOUT ANY WARRANTY; without 
cc even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
cc PARTICULAR PURPOSE.  See the GNU General Public License for more 
cc details. You should have received a copy of the GNU General Public 
cc License along with this program; 
cc if not, see <http://www.gnu.org/licenses/>.
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c    $Date: 2011-11-17 00:00:34 -0500 (Thu, 17 Nov 2011) $
c    $Revision: 2480 $
c
c
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c        this is the end of the debugging code and the beginning 
c        of the legendre expansion routines
c
ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
c
c
c
c        This file contains a set of subroutines for the handling 
c        of Legendre expansions. It contains 19 subroutines that are 
c        user-callable. Following is a brief description of these 
c        subroutines.
c
c   legeexps - constructs Legendre nodes, and  corresponding Gaussian
c        weights. Also constructs the matrix v converting the 
c         coefficients of a legendre expansion into its values at 
c         the n Gaussian nodes, and its inverse u, converting the
c         values of a function at n Gaussian nodes into the
c         coefficients of the corresponding Legendre series.
c
c   legepols - evaluates a bunch of Legendre polynomials
c         at the user-provided point

c   legeodev - evaluates at the point x a Legendre expansion
c        having only odd-numbered elements; this is a fairly 
c        efficient code, using external arrays that are 
c        precomputed
c
c   legeevev - evaluates at the point x a Legendre expansion
c        having only even-numbered elements; this is a fairly 
c        efficient code, using external arrays that are 
c        precomputed
c
c   legepeven - evaluates even-numbered Legendre polynomials 
c        of the argument x; this is a fairly efficient code, 
c        using external arrays that are precomputed
c
c   legepodd - evaluates odd-numbered Legendre polynomials 
c        of the argument x; this is a fairly efficient code, 
c        using external arrays that are precomputed
c
C   legefdeq - computes the value and the derivative of a
c        Legendre Q-expansion with coefficients coefs
C     at point X in interval (-1,1); please note that this is
c     the evil twin of the subroutine legefder, evaluating the
c     proper (P-function) Legendre expansion; this subroutine 
c        is not designed to be very efficient, but it does not
c        use any exdternally supplied arrays
c
c   legeq - calculates the values and derivatives of a bunch 
c        of Legendre Q-functions at the user-specified point 
c        x on the interval (-1,1)
c
c   legeqs - calculates the value and the derivative of a single
c        Legendre Q-function at the user-specified point 
c        x on the interval (-1,1)
c
c   legecfde - computes the value and the derivative of a Legendre 
c        expansion with complex coefficients at point X in interval 
c        [-1,1]; this subroutine is not designed to be very efficient, 
c        but it does not use any exdternally supplied arrays. This is
c        a complex version of the subroutine legefder.
c
c   legecfd2 - the same as legecfde, except it is designed to be 
c        fairly efficient; it uses externally supplied arrays
c        that are precomputed. This is a complex version of the 
c        subroutine legefde2.
c
c   legecva2 - the same as legecfd2, except it is does not evaluate
c        the derivative of the function
c
c
        subroutine legeexps(itype,n,x,u,v,whts)
        implicit real *8 (a-h,o-z)
        dimension x(1),whts(1),u(n,n),v(n,n)
c
c         this subroutine constructs the gaussiaqn nodes 
c         on the interval [-1,1], and the weights for the 
c         corresponding order n quadrature. it also constructs
c         the matrix v converting the coefficients
c         of a legendre expansion into its values at the n
c         gaussian nodes, and its inverse u, converting the
c         values of a function at n gaussian nodes into the
c         coefficients of the corresponding legendre series.
c         no attempt has been made to make this code efficient, 
c         but its speed is normally sufficient, and it is 
c         mercifully short.
c
c                 input parameters:
c
c  itype - the type of the calculation to be performed
c          itype=0 means that only the gaussian nodes are 
c                  to be constructed. 
c          itype=1 means that only the nodes and the weights 
c                  are to be constructed
c          itype=2 means that the nodes, the weights, and
c                  the matrices u, v are to be constructed
c  n - the number of gaussian nodes and weights to be generated
c  
c                 output parameters:
c
c  x - the order n gaussian nodes - computed independently
c          of the value of itype.
c  u - the n*n matrix converting the  values at of a polynomial of order
c         n-1 at n legendre nodes into the coefficients of its 
c         legendre expansion - computed only in itype=2
c  v - the n*n matrix converting the coefficients
c         of an n-term legendre expansion into its values at
c         n legendre nodes (note that v is the inverse of u)
c          - computed only in itype=2
c  whts - the corresponding quadrature weights - computed only 
c         if itype .ge. 1
c
c       . . . construct the nodes and the weights of the n-point gaussian 
c             quadrature
c
        ifwhts=0
        if(itype. gt. 0) ifwhts=1
        call legewhts(n,x,whts,ifwhts)
c
c       construct the matrix of values of the legendre polynomials
c       at these nodes        
c
        if(itype .ne. 2) return
        do 1400 i=1,n
c
        call legepols(x(i),n-1,u(1,i) )
 1400 continue
c
        do 1800 i=1,n
        do 1600 j=1,n
        v(i,j)=u(j,i)
 1600 continue
 1800 continue
c
c       now, v converts coefficients of a legendre expansion
c       into its values at the gaussian nodes. construct its 
c       inverse u, converting the values of a function at 
c       gaussian nodes into the coefficients of a legendre 
c       expansion of that function
c
        do 2800 i=1,n
        d=1
        d=d*(2*i-1)/2
        do 2600 j=1,n
        u(i,j)=v(j,i)*whts(j)*d
 2600 continue
 2800 continue
        return
        end
c
c
c
c
c
        subroutine legewhts(n,ts,whts,ifwhts)
        implicit real *8 (a-h,o-z)
        dimension ts(1),whts(1),ws2(1000),rats(1000)
c
c        this subroutine constructs the nodes and the
c        weights of the n-point gaussian quadrature on 
c        the interval [-1,1]
c
c                input parameters:
c
c  n - the number of nodes in the quadrature
c
c                output parameters:
c
c  ts - the nodes of the n-point gaussian quadrature
c  w - the weights of the n-point gaussian quadrature
c
c       . . . construct the array of initial approximations
c             to the roots of the n-th legendre polynomial
c
        eps=1.0d-14
        ZERO=0
        DONE=1
        pi=datan(done)*4
        h=pi/(2*n) 
        do 1200 i=1,n
        t=(2*i-1)*h
        ts(n-i+1)=dcos(t)
1200  CONTINUE
c
c         use newton to find all roots of the legendre polynomial
c
        ts(n/2+1)=0
        do 2000 i=1,n/2
c
        xk=ts(i)
        ifout=0
        deltold=1
        do 1400 k=1,10
        call legepol_sum(xk,n,pol,der,sum)
        delta=-pol/der
        xk=xk+delta
        if(abs(delta) .lt. eps) ifout=ifout+1
c
        if(ifout .eq. 3) goto 1600
 1400 continue
 1600 continue
        ts(i)=xk
        ts(n-i+1)=-xk
 2000 continue
c     
c        construct the weights via the orthogonality relation
c
        if(ifwhts .eq. 0) return
c
        do 2400 i=1,(n+1)/2
        call legepol_sum(ts(i),n,pol,der,sum)
        whts(i)=1/sum
        whts(n-i+1)=whts(i)
 2400 continue
c
        return
        end
c
c
c
c
c
        subroutine legepol_sum(x,n,pol,der,sum)
        implicit real *8 (a-h,o-z)
c
        done=1
        sum=0 
c
        pkm1=1
        pk=x
        sum=sum+pkm1**2 /2
        sum=sum+pk**2 *(1+done/2)
c
        pk=1
        pkp1=x
c
c        if n=0 or n=1 - exit
c
        if(n .ge. 2) goto 1200

        sum=0 
c
        pol=1
        der=0
        sum=sum+pol**2 /2
        if(n .eq. 0) return
c
        pol=x
        der=1
        sum=sum+pol**2*(1+done/2)
        return
 1200 continue
c
c       n is greater than 1. conduct recursion
c
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
        sum=sum+pkp1**2*(k+1+done/2)
 2000 continue
c
c        calculate the derivative
c
        pol=pkp1
        der=n*(x*pkp1-pk)/(x**2-1)
        return
        end
c
c
c
c
c
        subroutine legepols(x,n,pols)
        implicit real *8 (a-h,o-z)
        dimension pols(*)
c
        pkm1=1
        pk=x
c
        pk=1
        pkp1=x
c
c
c        if n=0 or n=1 - exit
c
        if(n .ge. 2) goto 1200
        pols(1)=1
        if(n .eq. 0) return
c
        pols(2)=x
        return
 1200 continue
c
        pols(1)=1
        pols(2)=x
c
c       n is greater than 2. conduct recursion
c
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
        pols(k+2)=pkp1
 2000 continue
c
        return
        end

c
c
c
c




        subroutine lematvec(a,x,y,n)
        implicit real *8 (a-h,o-z)
        dimension a(n,n),x(n),y(n)
c
        do 1400 i=1,n
        d=0
        do 1200 j=1,n
        d=d+a(j,i)*x(j)
 1200 continue
        y(i)=d
 1400 continue
        return
        end
c
c
c
c
c
        subroutine matmul(a,b,c,n)
        implicit real *8 (a-h,o-z)
        dimension a(n,n),b(n,n),c(n,n)
c
        do 2000 i=1,n
        do 1800 j=1,n
        d=0
        do 1600 k=1,n
        d=d+a(i,k)*b(k,j)
 1600 continue
        c(i,j)=d
 1800 continue
 2000 continue
        return
c
c
c
c
        entry matmua(a,b,c,n)
ccc          call prin2('in matmua, a=*',a,n**2)
ccc          call prin2('in matmua, b=*',b,n**2)
        do 3000 i=1,n
        do 2800 j=1,n
        d=0
        do 2600 k=1,n
        d=d+a(i,k)*b(j,k)
 2600 continue
        c(i,j)=d
 2800 continue
 3000 continue
ccc          call prin2('exiting, c=*',c,n**2)
        return
        end

c
c
c
c
c
        subroutine legeodev(x,nn,coefs,val,ninit,
     1      coepnm1,coepnp1,coexpnp1)
        implicit real *8 (a-h,o-z)
        dimension coepnm1(1),coepnp1(1),
     1            coexpnp1(1),coefs(*)
c
c
c       This subroutine evaluates at the point x a Legendre expansion
c       having only odd-numbered elements
c
c                  Input parameters:
c
c  x - point on the interval [-1,1] at which the Legendre expansion 
c       is to be evaluated
c  nn - order of the expansion to be evaluated
c  coefs - odd-numbered coefficients of the Legendre expansion
c       to be evaluated at the point x (nn/2+2 of them things)
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine will initialize the first ninit/2+2
c                  (or so) elements of each of the arrays  coepnm1,
c       coepnp1, coexpnp1. On the first call to this subroutine, ninit 
c       should be set to the maximum order nn for which this subroutine 
c       might have to be called; on subsequent calls, ninit should be 
c       set to 0. PLEASE NOTE THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEPODD. IF these arrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are input arrays only if ninit 
c       (see above) has been set to 0; otherwise, these are output arrays.
c 
c                  Output parameters:
c
c  val - the value at the point x of the Legendre expansion with 
c       coefficients coefs (see above) 
c    EXPLANATION: On exit from the subroutine, pols(1) = P_0 (x),
c       pols(2) = P_2(x), pols(3) =P_4(x),  etc.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are output parameters only if ninit 
c       (see above) has not been set to 0; otherwise, these are input 
c       parameters
c 
c       
        if(ninit .eq. 0) goto 1400
        done=1
        n=0
        i=0
c
        do 1200 nnn=2,ninit,2
c
        n=n+2
        i=i+1
c
        coepnm1(i)=-(5*n+7*(n*done)**2+2*(n*done)**3)
        coepnp1(i)=-(9+24*n+18*(n*done)**2+4*(n*done)**3)
        coexpnp1(i)=15+46*n+36*(n*done)**2+8*(n*done)**3
c
        d=(2+n*done)*(3+n*done)*(1+2*n*done)
        coepnm1(i)=coepnm1(i)/d
        coepnp1(i)=coepnp1(i)/d
        coexpnp1(i)=coexpnp1(i)/d
c
 1200 continue
c
 1400 continue
c
        x22=x**2
c
        pi=x
        pip1=x*(2.5d0*x22-1.5d0)
c
        val=coefs(1)*pi+coefs(2)*pip1

        do 2000 i=1,nn/2-2
c
        pip2 = coepnm1(i)*pi +
     1      (coepnp1(i)+coexpnp1(i)*x22)*pip1
c
        val=val+coefs(i+2)*pip2
c
        pi=pip1
        pip1=pip2


 2000 continue
c
        return
        end
c
c
c
c
c
        subroutine legeevev(x,nn,coefs,val,ninit,
     1      coepnm1,coepnp1,coexpnp1)
        implicit real *8 (a-h,o-z)
        dimension coepnm1(1),coepnp1(1),
     1            coexpnp1(1),coefs(*)
c
c
c       This subroutine evaluates at the point x a Legendre expansion
c       having only even-numbered elements
c
c                  Input parameters:
c
c  x - point on the interval [-1,1] at which the Legendre expansion 
c       is to be evaluated
c  nn - order of the expansion to be evaluated
c  coefs - even-numbered coefficients of the Legendre expansion
c       to be evaluated at the point x (nn/2+2 of them things)
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine will initialize the first ninit/2+2
c                  (or so) elements of each of the arrays  coepnm1,
c       coepnp1, coexpnp1. On the first call to this subroutine, ninit 
c       should be set to the maximum order nn for which this subroutine 
c       might have to be called; on subsequent calls, ninit should be 
c       set to 0. PLEASE NOTE THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEPEVEN. IF these aqrrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are input arrays only if ninit 
c       (see above) has been set to 0; otherwise, these are output arrays.
c 
c                  Output parameters:
c
c  val - the value at the point x of the Legendre expansion with 
c       coefficients coefs (see above) 
c    EXPLANATION: On exit from the subroutine, pols(1) = P_0 (x),
c       pols(2) = P_2(x), pols(3) =P_4(x),  etc.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are output parameters only if ninit 
c       (see above) has not been set to 0; otherwise, these are input 
c       parameters
c       
        if(ninit .eq. 0) goto 1400
c
        done=1
        n=-1
        i=0
        do 1200 nnn=1,ninit,2
c
        n=n+2
        i=i+1
c
        coepnm1(i)=-(5*n+7*(n*done)**2+2*(n*done)**3)
        coepnp1(i)=-(9+24*n+18*(n*done)**2+4*(n*done)**3)
        coexpnp1(i)=15+46*n+36*(n*done)**2+8*(n*done)**3
c
        d=(2+n*done)*(3+n*done)*(1+2*n*done)
        coepnm1(i)=coepnm1(i)/d
        coepnp1(i)=coepnp1(i)/d
        coexpnp1(i)=coexpnp1(i)/d
c
 1200 continue
c
 1400 continue
c
        x22=x**2
c
        pi=1
        pip1=1.5d0*x22-0.5d0
c
        val=coefs(1)+coefs(2)*pip1
c
c       n is greater than 2. conduct recursion
c
        do 2000 i=1,nn/2-2
c
        pip2 = coepnm1(i)*pi +
     1      (coepnp1(i)+coexpnp1(i)*x22) *pip1
        val=val+coefs(i+2)*pip2
c
        pi=pip1
        pip1=pip2
c
 2000 continue
c
        return
        end
c
c
c
c
c
        subroutine legepeven(x,nn,pols,ninit,
     1      coepnm1,coepnp1,coexpnp1)
        implicit real *8 (a-h,o-z)
        dimension pols(*),coepnm1(1),coepnp1(1),
     1            coexpnp1(1)
c
c       This subroutine evaluates even-numbered Legendre polynomials 
c       of the argument x, up to order nn+1
c
c                  Input parameters:
c
c  x - the argument for which the Legendre polynomials are 
c       to be evaluated
c  nn - the maximum order for which the Legendre polynomials are
c       to be evaluated
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine ill initialize the first ninit/2+2
c                  (or so) elements of each of the arrays  coepnm1,
c       coepnp1, coexpnp1. On the first call to this subroutine, ninit 
c       should be set to the maximum order nn for which this subroutine 
c       might have to be called; on subsequent calls, ninit should be 
c       set to 0.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are input arrays only if ninit 
c       (see above) has been set to 0; otherwise, these are output arrays.
c       PLEASE NOTE THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEEVEV. IF these aqrrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c 
c                  Output parameters:
c
c  pols - even-numbered Legendre polynomials of the input parameter x
c         (nn/2+2 of them things)
c    EXPLANATION: On exit from the subroutine, pols(1) = P_0 (x),
c       pols(2) = P_2(x), pols(3) =P_4(x),  etc.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are output parameters only if ninit 
c       (see above) has not been set to 0; otherwise, these are input 
c       parameters. PLEASE NOTE THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEEVEV. If these arrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c 
c       
        if(ninit .eq. 0) goto 1400
c
        done=1
        n=-1
        i=0
        do 1200 nnn=1,ninit,2
c
        n=n+2
        i=i+1
c
        coepnm1(i)=-(5*n+7*(n*done)**2+2*(n*done)**3)
        coepnp1(i)=-(9+24*n+18*(n*done)**2+4*(n*done)**3)
        coexpnp1(i)=15+46*n+36*(n*done)**2+8*(n*done)**3
c
        d=(2+n*done)*(3+n*done)*(1+2*n*done)
        coepnm1(i)=coepnm1(i)/d
        coepnp1(i)=coepnp1(i)/d
        coexpnp1(i)=coexpnp1(i)/d
c
 1200 continue
c
 1400 continue
c
        x22=x**2
c
        pols(1)=1
        pols(2)=1.5d0*x22-0.5d0
c
c       n is greater than 2. conduct recursion
c
        do 2000 i=1,nn/2
c
        pols(i+2) = coepnm1(i)*pols(i) +
     1      (coepnp1(i)+coexpnp1(i)*x22) *pols(i+1)
c
 2000 continue
c
        return
        end
c
c
c
c
c
        subroutine legepodd(x,nn,pols,ninit,
     1      coepnm1,coepnp1,coexpnp1)
        implicit real *8 (a-h,o-z)
        dimension pols(*),coepnm1(1),coepnp1(1),
     1            coexpnp1(1)
c
c       This subroutine evaluates odd-numbered Legendre polynomials 
c       of the argument x, up to order nn+1
c
c                  Input parameters:
c
c  x - the argument for which the Legendre polynomials are 
c       to be evaluated
c  nn - the maximum order for which the Legendre polynomials are
c       to be evaluated
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine will initialize the first ninit/2+2
c                  (or so) elements of each of the arrays  coepnm1,
c       coepnp1, coexpnp1. On the first call to this subroutine, ninit 
c       should be set to the maximum order nn for which this subroutine 
c       might have to be called; on subsequent calls, ninit should be 
c       set to 0.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are input arrays only if ninit 
c       (see above) has been set to 0; otherwise, these are output arrays.
c 
c                  Output parameters:
c
c  pols - the odd-numbered Legendre polynomials of the input parameter x
c         (nn/2+2 of them things)
c    EXPLANATION: On exit from the subroutine, pols(1) = P_1(x),
c       pols(2) = P_3(x), pols(3) = P_5 (x), etc.
c  coepnm1,coepnp1,coexpnp1 - should be nn/2+4 real *8 elements long 
c       each. Please note that these are output parameters only if ninit 
c       (see above) has not been set to 0; otherwise, these are input 
c       parameters. PLEASE NOTE THAT THAT THESE ARRAYS USED BY THIS 
c       SUBROUTINE ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES 
c       SUSED BY THE UBROUTINE LEGEODEV. IF these arrays have been 
c       initialized by one of these two subroutines, they do not need 
c       to be initialized by the other one.
c       
        if(ninit .eq. 0) goto 1400
        done=1
        n=0
        i=0
c
        do 1200 nnn=2,ninit,2
c
        n=n+2
        i=i+1
c
        coepnm1(i)=-(5*n+7*(n*done)**2+2*(n*done)**3)
        coepnp1(i)=-(9+24*n+18*(n*done)**2+4*(n*done)**3)
        coexpnp1(i)=15+46*n+36*(n*done)**2+8*(n*done)**3
c
        d=(2+n*done)*(3+n*done)*(1+2*n*done)
        coepnm1(i)=coepnm1(i)/d
        coepnp1(i)=coepnp1(i)/d
        coexpnp1(i)=coexpnp1(i)/d
c
 1200 continue
c
 1400 continue
c
        x22=x**2
c
        pols(1)=x
        pols(2)=x*(2.5d0*x22-1.5d0)
c
        do 2000 i=1,nn/2
c
        pols(i+2) = coepnm1(i)*pols(i) +
     1      (coepnp1(i)+coexpnp1(i)*x22)*pols(i+1)
c
 2000 continue
c
        return
        end
c
c
c
c
c
        subroutine legefdeq(x,val,der,coefs,n)
        implicit real *8 (a-h,o-z)
        dimension coefs(*)
C
C     This subroutine computes the value and the derivative
c     of a Legendre Q-expansion with coefficients coefs
C     at point X in interval (-1,1); please note that this is
c     the evil twin of the subroutine legefder, evaluating the
c     proper (P-function) Legendre expansion
c
c                input parameters:
c
C  X = evaluation point
C  coefs = expansion coefficients
C  N  = order of expansion 
c
c   IMPORTANT NOTE: n is {\bf the order of the expansion, which is
c         one less than the number of terms in the expansion!!}
c
c                output parameters:
c
C     VAL = computed value
C     der = computed value of the derivative
C
c
        val=0
        der=0
c
        d= log( (1+x) /(1-x) ) /2
        pkm1=d
        pk=d*x-1
c
        pk=d
        pkp1=d*x-1

        derk=(1/(1+x)+1/(1-x)) /2
        derkp1=d + derk *x 
c
        val=coefs(1)*pk+coefs(2)*pkp1
        der=coefs(1)*derk+coefs(2)*derkp1
c
c        if n=0 or n=1 - exit
c
        if(n .ge. 2) goto 1200
c
        if(n .eq. 0) return
c
        return
 1200 continue
c
c       n is greater than 2. conduct recursion
c
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
c
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
c
        derkm1=derk
        derk=derkp1
c
        derkp1= ( (2*k+1)*pk+(2*k+1)*x*derk - k*derkm1 )/(k+1)
c
        val=val+coefs(k+2)*pkp1
        der=der+coefs(k+2)*derkp1
c
 2000 continue
c
        return
        end
c
c
c
c
c
        subroutine legeqs(x,n,pols,ders)
        implicit real *8 (a-h,o-z)
        dimension pols(*),ders(*)
c
c       This subroutine calculates the values and derivatives of 
c       a bunch of Legendre Q-functions at the user-specified point 
c       x on the interval (-1,1)
c
c                     Input parameters:
c
c  x - the point on the interval [-1,1] where the Q-functions and 
c       their derivatives are to be evaluated
c  n - the highest order for which the functions are to be evaluated
c  
c                     Output parameters:
c
c  pols - the values of the Q-functions (the evil twins of the 
c       Legeendre polynomials) at the point x (n+1 of them things)
c  ders - the derivatives of the Q-functions (the evil twins of the 
c       Legeendre polynomials) at the point x (n+1 of them things)
c  
c
        d= log( (1+x) /(1-x) ) /2
        pkm1=d
        pk=d*x-1
c
        pk=d
        pkp1=d*x-1

        derk=(1/(1+x)+1/(1-x)) /2
        derkp1=d + derk *x 
c
c        if n=0 or n=1 - exit
c
        if(n .ge. 2) goto 1200
        pols(1)=pk
        ders(1)=derk
        if(n .eq. 0) return
c
        pols(2)=pkp1
        ders(2)=derkp1
        return
 1200 continue
c
        pols(1)=pk
        pols(2)=pkp1
c
c       n is greater than 2. conduct recursion
c
        ders(1)=derk
        ders(2)=derkp1
c
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
c
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
        pols(k+2)=pkp1
c
        derkm1=derk
        derk=derkp1
c
        derkp1= ( (2*k+1)*pk+(2*k+1)*x*derk - k*derkm1 )/(k+1)
        ders(k+2)=derkp1
 2000 continue
c
        return
        end
c
c
c
c
c


        subroutine legeq(x,n,pol,der)
        implicit real *8 (a-h,o-z)
c
c       This subroutine calculates the value and derivative of 
c       a Legendre Q-function at the user-specified point 
c       x on the interval (-1,1)
c
c
c                     Input parameters:
c
c  x - the point on the interval [-1,1] where the Q-functions and 
c       their derivatives are to be evaluated
c  n - the order for which the function is to be evaluated
c  
c                     Output parameters:
c
c  pol - the value of the n-th Q-function (the evil twin of the 
c       Legeendre polynomial) at the point x 
c  ders - the derivatives of the Q-function at the point x 
c  
c
        d= log( (1+x) /(1-x) ) /2
        pk=d
        pkp1=d*x-1
c
c        if n=0 or n=1 - exit
c
        if(n .ge. 2) goto 1200
        pol=d

        der=(1/(1+x)+1/(1-x)) /2

        if(n .eq. 0) return
c
        pol=pkp1
        der=d + der *x 
        return
 1200 continue
c
c       n is greater than 1. conduct recursion
c
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
 2000 continue
c
c        calculate the derivative
c
        pol=pkp1
        der=n*(x*pkp1-pk)/(x**2-1)
        return
        end

c
c
c
c
c
      SUBROUTINE legecFDE(X,VAL,der,PEXP,N)
      IMPLICIT REAL *8 (A-H,O-Z)
      complex *16 PEXP(*),val,der
C
C     This subroutine computes the value and the derivative
c     of a gaussian expansion with complex coefficients PEXP
C     at point X in interval [-1,1].
c
c                input parameters:
c
C     X = evaluation point
C     PEXP = expansion coefficients
C     N  = order of expansion 
c   IMPORTANT NOTE: n is {\bf the order of the expansion, which is
c         one less than the number of terms in the expansion!!}
c
c                output parameters:
c
C     VAL = computed value
C     der = computed value of the derivative
C
C
        done=1
        pjm2=1
        pjm1=x
        derjm2=0
        derjm1=1
c
        val=pexp(1)*pjm2+pexp(2)*pjm1      
        der=pexp(2)
c
        DO 600 J = 2,N
c
        pj= ( (2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
        val=val+pexp(j+1)*pj
c
        derj=(2*j-1)*(pjm1+x*derjm1)-(j-1)*derjm2
c
        derj=derj/j
        der=der+pexp(j+1)*derj
c 
        pjm2=pjm1
        pjm1=pj
        derjm2=derjm1
        derjm1=derj
 600   CONTINUE
c
      RETURN
      END
c
c
c
c
c
      SUBROUTINE legecFD2(X,VAL,der,PEXP,N,
     1    pjcoefs1,pjcoefs2,ninit)
      IMPLICIT REAL *8 (A-H,O-Z)
      REAL *8 pjcoefs1(1),pjcoefs2(1)
      complex *16 PEXP(*),val,der
c
C     This subroutine computes the value and the derivative
c     of a Legendre expansion with complex coefficients PEXP
C     at point X in interval [-1,1].
c
c                input parameters:
c
C  X - evaluation point
C  PEXP - expansion coefficients
C  N  - order of expansion 
c  pjcoefs1, pjcoefs2 - two arrays precomputed on a previous call 
c      on a previous call to this subroutine. Please note that this
c      is only an input parameter if the parameter ninit (see below) 
c      has been set to 0; otherwise, these are output parameters
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine will initialize the first ninit
c       elements of each of the arrays pjcoefs1, pjcoefs2. On the first 
c       call to this subroutine, ninit should be set to the maximum 
c       order n for which this subroutine might have to be called; 
c       on subsequent calls, ninit should be set to 0. PLEASE NOTE 
c       THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEEXE2. If these arrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c
c   IMPORTANT NOTE: n is {\bf the order of the expansion, which is
c         one less than the number of terms in the expansion!!}
c
c                output parameters:
c
C  VAL - computed value
C  der - computed value of the derivative
C
        if(ninit .eq. 0) goto 1400
c
        done=1
        do 1200 j=2,ninit
c
        pjcoefs1(j)=(2*j-done)/j
        pjcoefs2(j)=-(j-done)/j
c
 1200 continue
c 
        ifcalled=1
 1400 continue
c
        pjm2=1
        pjm1=x
        derjm2=0
        derjm1=1
c
        val=pexp(1)*pjm2+pexp(2)*pjm1      
        der=pexp(2)
c
        DO 1600 J = 2,N
c
cccc        pj= ( (2*j-1)*x*pjm1-(j-1)*pjm2 ) / j
c
        pj= pjcoefs1(j)*x*pjm1+pjcoefs2(j)*pjm2


        val=val+pexp(j+1)*pj
c
cccc        derj=(2*j-1)*(pjm1+x*derjm1)-(j-1)*derjm2
        derj=pjcoefs1(j)*(pjm1+x*derjm1)+pjcoefs2(j)*derjm2

ccc         call prin2('derj=*',derj,1)


cccc        derj=derj/j
        der=der+pexp(j+1)*derj
c 
        pjm2=pjm1
        pjm1=pj
        derjm2=derjm1
        derjm1=derj
 1600   CONTINUE
c
      RETURN
      END
c
c
c
c
c
      SUBROUTINE legecva2(X,VAL,PEXP,N,
     1    pjcoefs1,pjcoefs2,ninit)
      IMPLICIT REAL *8 (A-H,O-Z)
      REAL *8 pjcoefs1(1),pjcoefs2(1)
      complex *16 PEXP(*),val
c
C     This subroutine computes the value of a Legendre expansion 
c     with complex coefficients PEXP at point X in interval [-1,1].
c
c                input parameters:
c
C  X - evaluation point
C  PEXP - expansion coefficients
C  N  - order of expansion 
c  pjcoefs1, pjcoefs2 - two arrays precomputed on a previous call 
c      on a previous call to this subroutine. Please note that this
c      is only an input parameter if the parameter ninit (see below) 
c      has been set to 0; otherwise, these are output parameters
c  ninit - tells the subroutine whether and to what maximum order the 
c       arrays coepnm1,coepnp1,coexpnp1 should be initialized.
c     EXPLANATION: The subroutine will initialize the first ninit
c       elements of each of the arrays pjcoefs1, pjcoefs2. On the first 
c       call to this subroutine, ninit should be set to the maximum 
c       order n for which this subroutine might have to be called; 
c       on subsequent calls, ninit should be set to 0. PLEASE NOTE 
c       THAT THAT THESE ARRAYS USED BY THIS SUBROUTINE
c       ARE IDENTICAL TO THE ARRAYS WITH THE SAME NAMES USED BY THE 
c       SUBROUTINE LEGEEXE2. If these arrays have been initialized
c       by one of these two subroutines, they do not need to be 
c       initialized by the other one.
c
c   IMPORTANT NOTE: n is {\bf the order of the expansion, which is
c         one less than the number of terms in the expansion!!}
c
c                output parameters:
c
C  VAL - computed value
C
        if(ninit .eq. 0) goto 1400
c
        done=1
        do 1200 j=2,ninit
c
        pjcoefs1(j)=(2*j-done)/j
        pjcoefs2(j)=-(j-done)/j
c
 1200 continue
c 
        ifcalled=1
 1400 continue
c
        pjm2=1
        pjm1=x
c
        val=pexp(1)*pjm2+pexp(2)*pjm1      
c
        DO 1600 J = 2,N
c
        pj= pjcoefs1(j)*x*pjm1+pjcoefs2(j)*pjm2
        val=val+pexp(j+1)*pj
c
        pjm2=pjm1
        pjm1=pj
 1600   CONTINUE
c
      RETURN
      END


