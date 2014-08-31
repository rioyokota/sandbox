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
