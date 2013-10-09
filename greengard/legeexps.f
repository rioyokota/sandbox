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
        ifwhts=0
        if(itype. gt. 0) ifwhts=1
        call legewhts(n,x,whts,ifwhts)
c       construct the matrix of values of the legendre polynomials
c       at these nodes        
        if(itype .ne. 2) return
        do 1400 i=1,n
        call legepols(x(i),n-1,u(1,i) )
 1400 continue
        do 1800 i=1,n
        do 1600 j=1,n
        v(i,j)=u(j,i)
 1600 continue
 1800 continue
c       now, v converts coefficients of a legendre expansion
c       into its values at the gaussian nodes. construct its 
c       inverse u, converting the values of a function at 
c       gaussian nodes into the coefficients of a legendre 
c       expansion of that function
        do 2800 i=1,n
        d=1
        d=d*(2*i-1)/2
        do 2600 j=1,n
        u(i,j)=v(j,i)*whts(j)*d
 2600 continue
 2800 continue
        return
        end

        subroutine legewhts(n,ts,whts,ifwhts)
        implicit real *8 (a-h,o-z)
        dimension ts(1),whts(1),ws2(1000),rats(1000)
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
        eps=1.0d-14
        ZERO=0
        DONE=1
        pi=datan(done)*4
        h=pi/(2*n) 
        do 1200 i=1,n
        t=(2*i-1)*h
        ts(n-i+1)=dcos(t)
1200  CONTINUE
c         use newton to find all roots of the legendre polynomial
        ts(n/2+1)=0
        do 2000 i=1,n/2
        xk=ts(i)
        ifout=0
        deltold=1
        do 1400 k=1,10
        call legepol_sum(xk,n,pol,der,sum)
        delta=-pol/der
        xk=xk+delta
        if(abs(delta) .lt. eps) ifout=ifout+1
        if(ifout .eq. 3) goto 1600
 1400 continue
 1600 continue
        ts(i)=xk
        ts(n-i+1)=-xk
 2000 continue
c        construct the weights via the orthogonality relation
        if(ifwhts .eq. 0) return
        do 2400 i=1,(n+1)/2
        call legepol_sum(ts(i),n,pol,der,sum)
        whts(i)=1/sum
        whts(n-i+1)=whts(i)
 2400 continue
        return
        end

        subroutine legepol_sum(x,n,pol,der,sum)
        implicit real *8 (a-h,o-z)
        done=1
        sum=0 
        pkm1=1
        pk=x
        sum=sum+pkm1**2 /2
        sum=sum+pk**2 *(1+done/2)
        pk=1
        pkp1=x
        if(n .ge. 2) goto 1200
        sum=0 
        pol=1
        der=0
        sum=sum+pol**2 /2
        if(n .eq. 0) return
        pol=x
        der=1
        sum=sum+pol**2*(1+done/2)
        return
 1200 continue
c       n is greater than 1. conduct recursion
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
        sum=sum+pkp1**2*(k+1+done/2)
 2000 continue
c        calculate the derivative
        pol=pkp1
        der=n*(x*pkp1-pk)/(x**2-1)
        return
        end

        subroutine legepols(x,n,pols)
        implicit real *8 (a-h,o-z)
        dimension pols(*)
        pkm1=1
        pk=x
        pk=1
        pkp1=x
c       if n=0 or n=1 - exit
        if(n .ge. 2) goto 1200
        pols(1)=1
        if(n .eq. 0) return
        pols(2)=x
        return
 1200 continue
        pols(1)=1
        pols(2)=x
c       n is greater than 2. conduct recursion
        do 2000 k=1,n-1
        pkm1=pk
        pk=pkp1
        pkp1=( (2*k+1)*x*pk-k*pkm1 )/(k+1)
        pols(k+2)=pkp1
 2000 continue
        return
        end
