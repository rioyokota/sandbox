        program main
        implicit real *8 (a-h,o-z)
        real *8     source(3,1 000 000)
        complex *16 charge(1 000 000)
        complex *16 pot(1 000 000)
        complex *16 fld(3,1 000 000)
        complex *16 pot2,fld2(3)
        complex *16 ptemp,ftemp(3)
        complex *16 ima
        complex *16 zk
        data ima/(0.0d0,1.0d0)/
        pi=4*atan(1.0d0)
        nsource= 100000
        zk = 1.0d0 + ima*0.1d0
        call random_number(source)
        print*,'nsource=',nsource
        iprec=0
        print*,'iprec  =',iprec
        ifcharge=1
        do i=1,nsource
           charge(i)=source(1,i)+ima*source(2,i)
        enddo
c FMM
        t1=omp_get_wtime()
        call fmm(ier,iprec, zk,
     1     nsource,source,ifcharge,charge,
     1     pot,fld)
        t2=omp_get_wtime()
        print*,'FMM    =',t2-t1
c Direct
        pdiff = 0
        pnorm = 0
        fdiff = 0
        fnorm = 0
        t1=omp_get_wtime()
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,ptemp,ftemp,pot2,fld2)
C$OMP$REDUCTION(+:pdiff,pnorm,fdiff,fnorm)
        do i=1,min(nsource,100)
           pot2=0
           do j=1,3
              fld2(j)=0
           enddo
           do j=1,nsource       
              if( i .eq. j ) cycle
                 call P2P(source(1,j),charge(j),
     1              source(1,i),zk,ptemp,ftemp)
                 pot2=pot2+ptemp
                 fld2(1)=fld2(1)+ftemp(1)
                 fld2(2)=fld2(2)+ftemp(2)
                 fld2(3)=fld2(3)+ftemp(3)
           enddo
           pdiff = pdiff+abs(pot(i)-pot2)**2
           pnorm = pnorm+abs(pot2)**2
           fdiff = fdiff+abs(fld(1,i)-fld2(1))**2
     1          +abs(fld(2,i)-fld2(2))**2
     1          +abs(fld(3,i)-fld2(3))**2
           fnorm = fnorm+abs(fld2(1))**2
     1          +abs(fld2(2))**2
     1          +abs(fld2(3))**2
        enddo
C$OMP END PARALLEL DO
        t2=omp_get_wtime()
        print*,'Direct =',t2-t1
        print*,'Err pot=',sqrt(pdiff/pnorm)
        print*,'Err acc=',sqrt(fdiff/fnorm)
        stop
        end
