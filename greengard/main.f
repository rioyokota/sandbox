        program main
        implicit real *8 (a-h,o-z)
        real *8     source(3,1 000 000)
        complex *16 charge(1 000 000)
        complex *16 pot(1 000 000)
        complex *16 fld(3,1 000 000)
        complex *16 pot2(1 000 000)
        complex *16 fld2(3,1 000 000)
        complex *16 ptemp,ftemp(3)
        complex *16 ima
        complex *16 zk
        data ima/(0.0d0,1.0d0)/
        pi=4*atan(1.0d0)
        call prini(6,13)
        nsource= 100000
        zk = 1.0d0 + ima*0.1d0
        call random_number(source)
        print*,'nsource=',nsource
        iprec=0
        print*,'iprec  =',iprec
        ifpot=1
        iffld=1
        ifcharge=1
        do i=1,nsource
           charge(i)=source(1,i)+ima*source(2,i)
        enddo
c FMM
        t1=omp_get_wtime()
        call hfmm3dparttarg(ier,iprec, zk,
     $     nsource,source,ifcharge,charge,
     $     ifpot,pot,iffld,fld)
        t2=omp_get_wtime()
        print*,'FMM    =',t2-t1
c Direct
        m=min(nsource,100)
        do i=1,nsource
           pot2(i)=0
           fld2(1,i)=0
           fld2(2,i)=0
           fld2(3,i)=0
        enddo
        t1=omp_get_wtime()
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,ptemp,ftemp) 
        do j=1,m
           do i=1,nsource       
              if( i .eq. j ) cycle
                 call hpotfld3d(iffld,source(1,i),charge(i),
     $              source(1,j),zk,ptemp,ftemp)
                 pot2(j)=pot2(j)+ptemp
                 fld2(1,j)=fld2(1,j)+ftemp(1)
                 fld2(2,j)=fld2(2,j)+ftemp(2)
                 fld2(3,j)=fld2(3,j)+ftemp(3)
           enddo
        enddo
C$OMP END PARALLEL DO
        t2=omp_get_wtime()
        print*,'Direct =',t2-t1
c Error
        call h3derror(pot,pot2,m,aerr,rerr)
        print*,'Err pot=',rerr
        call h3derror(fld,fld2,3*m,aerr,rerr)
        print*,'Err acc=',rerr
        stop
        end
c
        subroutine h3derror(pot1,pot2,n,ae,re)
        implicit real *8 (a-h,o-z)
        complex *16 pot1(n),pot2(n)
        d=0
        a=0
        do i=1,n
           d=d+abs(pot1(i)-pot2(i))**2
           a=a+abs(pot1(i))**2
        enddo
        d=d/n
        d=sqrt(d)
        a=a/n
        a=sqrt(a)
        ae=d
        re=d/a
        return
        end
