        program testhelm
        implicit real *8 (a-h,o-z)
        real *8     source(3,1 000 000)
        real *8     target(3,2 000 000)
        complex *16 charge(1 000 000)
        complex *16 dipstr(1 000 000)
        real *8     dipvec(3,1 000 000)
        complex *16 pot(1 000 000)
        complex *16 fld(3,1 000 000)
        complex *16 pot2(1 000 000)
        complex *16 fld2(3,1 000 000)
        complex *16 pottarg(2 000 000)
        complex *16 fldtarg(3,2 000 000)
        complex *16 ptemp,ftemp(3)
        complex *16 ima
        complex *16 zk
        data ima/(0.0d0,1.0d0)/
        done=1
        pi=4*atan(done)
        call prini(6,13)
        nsource= 100000
        ntarget=nsource
        zk = 1.0d0 + ima*0.1d0
        call random_number(source)
        call random_number(target)
 
        print*,'ntarget=',ntarget
        iprec=0
        print*,'iprec  =',iprec
c     set source type flags and output flags
        ifpot=1
        iffld=1
c
        ifcharge=1
        ifdipole=0
c
        ifpottarg=1
        iffldtarg=1
        do i=1,nsource
           charge(i)=source(1,i)+ima*source(2,i)
        enddo
c
        t1=omp_get_wtime()
c       
c     call FMM3D routine for sources and targets
c
        call hfmm3dparttarg(ier,iprec, zk,
     $     nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,
     $     ifpot,pot,iffld,fld,ntarget,target,
     $     ifpottarg,pottarg,iffldtarg,fldtarg)
c       
c     get time for FMM call
c
        t2=omp_get_wtime()
c       
c       
        print*,'FMM    =',t2-t1
c       
c     call direct calculation with subset of points to assess accuracy
c
        m=min(nsource,100)
c
c     ifprint=0 suppresses printing of source locations
c     ifprint=1 turns on printing of source locations
c
c        ifprint=0
c        if (ifprint .eq. 1) then
c        call prin2('source=*',source,3*nsource)
c        endif
c
c     ifprint=0 suppresses printing of potentials and fields
c     ifprint=1 turns on printing of potentials and fields
c
c        ifprint=0
c        if (ifprint .eq. 1) then
c           if( ifpot.eq.1 ) call prin2('after fmm, pot=*',pot,2*m)
c           if( iffld.eq.1 ) call prin2('after fmm, fld=*',fld,3*2*m)
c        endif
c
c       for direct calculation, initialize pot2,fld2 arrays to zero.
c
        do i=1,nsource
           if (ifpot .eq. 1) pot2(i)=0
           if (iffld .eq. 1) then
              fld2(1,i)=0
              fld2(2,i)=0
              fld2(3,i)=0
           endif
        enddo
c        
        t1=omp_get_wtime()
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,ptemp,ftemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do j=1,m
           do i=1,nsource       
              if( i .eq. j ) cycle
              if( ifcharge .eq. 1 ) then
                 call hpotfld3d(iffld,source(1,i),charge(i),
     $              source(1,j),zk,ptemp,ftemp)
                 if (ifpot .eq. 1) pot2(j)=pot2(j)+ptemp
                 if (iffld .eq. 1) then
                    fld2(1,j)=fld2(1,j)+ftemp(1)
                    fld2(2,j)=fld2(2,j)+ftemp(2)
                    fld2(3,j)=fld2(3,j)+ftemp(3)
                 endif
              endif
              if (ifdipole .eq. 1) then
                 call hpotfld3d_dp(iffld,source(1,i),
     $              dipstr(i),dipvec(1,i),
     $              source(1,j),zk,ptemp,ftemp)
                 if (ifpot .eq. 1) pot2(j)=pot2(j)+ptemp
                 if (iffld .eq. 1) then
                    fld2(1,j)=fld2(1,j)+ftemp(1)
                    fld2(2,j)=fld2(2,j)+ftemp(2)
                    fld2(3,j)=fld2(3,j)+ftemp(3)
                 endif
              endif
           enddo
        enddo
C$OMP END PARALLEL DO
c
        t2=omp_get_wtime()
c
c       ifprint=1 turns on printing of first m values of potential and field
c
c        if (ifprint .eq. 1) then
c           if( ifpot.eq.1 ) call prin2('directly, pot=*',pot2,2*m)
c           if( iffld.eq.1 ) call prin2('directly, fld=*',fld2,3*2*m)
c        endif
c
        print*,'Direct =',(t2-t1)*dble(nsource)/dble(m)
c       
        if (ifpot .eq. 1)  then
           call h3derror(pot,pot2,m,aerr,rerr)
           print*,'Err pot=',rerr
        endif
c
        if (iffld .eq. 1) then
           call h3derror(fld,fld2,3*m,aerr,rerr)
           print*,'Err acc=',rerr
        endif
c       
c
        do i=1,ntarget
           if (ifpottarg .eq. 1) pot2(i)=0
           if (iffldtarg .eq. 1) then
              fld2(1,i)=0
              fld2(2,i)=0
              fld2(3,i)=0
           endif
        enddo
c        
        t1=omp_get_wtime()
c
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,ptemp,ftemp) 
cccC$OMP$SCHEDULE(DYNAMIC)
cccC$OMP$NUM_THREADS(4) 
        do j=1,m
        do i=1,nsource        
           if( ifcharge .eq. 1 ) then
              call hpotfld3d(iffldtarg,
     $           source(1,i),charge(i),target(1,j),zk,
     $           ptemp,ftemp)
              if (ifpottarg .eq. 1) pot2(j)=pot2(j)+ptemp
              if (iffldtarg .eq. 1) then
                 fld2(1,j)=fld2(1,j)+ftemp(1)
                 fld2(2,j)=fld2(2,j)+ftemp(2)
                 fld2(3,j)=fld2(3,j)+ftemp(3)
              endif
           endif
           if (ifdipole .eq. 1) then
              call hpotfld3d_dp(iffldtarg,
     $           source(1,i),dipstr(i),dipvec(1,i),
     $           target(1,j),zk,ptemp,ftemp)
              if (ifpottarg .eq. 1) pot2(j)=pot2(j)+ptemp
              if (iffldtarg .eq. 1) then
                 fld2(1,j)=fld2(1,j)+ftemp(1)
                 fld2(2,j)=fld2(2,j)+ftemp(2)
                 fld2(3,j)=fld2(3,j)+ftemp(3)
              endif
           endif
c
        enddo
        enddo
C$OMP END PARALLEL DO
c
        t2=omp_get_wtime()
        stop
        end
c
c
c
c
        subroutine h3derror(pot1,pot2,n,ae,re)
        implicit real *8 (a-h,o-z)
c
c       evaluate absolute and relative errors
c
        complex *16 pot1(n),pot2(n)
c
        d=0
        a=0
c       
        do i=1,n
           d=d+abs(pot1(i)-pot2(i))**2
           a=a+abs(pot1(i))**2
        enddo
c       
        d=d/n
        d=sqrt(d)
        a=a/n
        a=sqrt(a)
c       
        ae=d
        re=d/a
c       
        return
        end
c
c
c
