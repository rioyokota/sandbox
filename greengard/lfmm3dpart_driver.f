cc Copyright (C) 2009-2012: Leslie Greengard and Zydrunas Gimbutas
cc Contact: greengard@cims.nyu.edu
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
c     This is the first release of the FMM3D library, together with
c     associated subroutines, which computes N-body interactions
c     governed by the Laplace or Helmholtz equations.
c  
c
        program testlap
        implicit real *8 (a-h,o-z)
        real *8     source(3,1 000 000)
        complex *16 charge(1 000 000)
        complex *16 dipstr(1 000 000)
        real *8     dipvec(3,1 000 000)
        complex *16 pot(1 000 000)
        complex *16 fld(3,1 000 000)
c       
        complex *16 pot2(1 000 000)
        complex *16 fld2(3,1 000 000)
c       
        real *8     target(3,2 000 000)
        complex *16 pottarg(2 000 000)
        complex *16 fldtarg(3,2 000 000)
c
        complex *16 ptemp,ftemp(3)
c       
        complex *16 ima
        data ima/(0.0d0,1.0d0)/
c
        done=1
        pi=4*atan(done)
c
c     Initialize simple printing routines. The parameters to prini
c     define output file numbers using standard Fortran conventions.
c
c     Calling prini(6,13) causes printing to the screen and to 
c     file fort.13.     
c
        call prini(6,13)
c
        nsource= 1000000
c
c     construct randomly located charge distribution on a unit sphere
c 
        d=hkrand(0)
        do i=1,nsource
c           theta=hkrand(0)*pi
c           phi=hkrand(0)*2*pi
c           source(1,i)=.5d0*cos(phi)*sin(theta)
c           source(2,i)=.5d0*sin(phi)*sin(theta)
c           source(3,i)=.5d0*cos(theta)
           source(1,i)=hkrand(0)
           source(2,i)=hkrand(0)
           source(3,i)=hkrand(0)
        enddo
c
c     construct target distribution on a target unit sphere 
c
        ntarget=nsource
        do i=1,ntarget
c           theta=hkrand(0)*pi
c           phi=hkrand(0)*2*pi
c           target(1,i)=.5d0*cos(phi)*sin(theta)
c           target(2,i)=.5d0*sin(phi)*sin(theta)
c           target(3,i)=.5d0*cos(theta)
           target(1,i)=hkrand(0)
           target(2,i)=hkrand(0)
           target(3,i)=hkrand(0)
        enddo
c
        print*,'ntarget=',ntarget
        iprec=0
        print*,'iprec  =',iprec
c       
c     set source type flags and output flags
c
        ifpot=1
        iffld=1
c
        ifcharge=1
        ifdipole=0
c
        ifpottarg=1
        iffldtarg=1
        if (ifcharge .eq. 1 ) then
           do i=1,nsource
c              charge(i)=hkrand(0) + ima*hkrand(0)
              charge(i)=1.0 / nsource
           enddo
        endif
        if (ifdipole .eq. 1) then
           do i=1,nsource
              dipstr(i)=hkrand(0) + ima*hkrand(0)
              dipvec(1,i)=hkrand(0)
              dipvec(2,i)=hkrand(0)
              dipvec(3,i)=hkrand(0)
           enddo
        endif
        t1=omp_get_wtime()
        call lfmm3dparttarg(ier,iprec,
     $     nsource,source,ifcharge,charge,ifdipole,dipstr,dipvec,
     $     ifpot,pot,iffld,fld,ntarget,target,
     $     ifpottarg,pottarg,iffldtarg,fldtarg)
        t2=omp_get_wtime()
        print*,'FMM    =',t2-t1
        m=min(nsource,100)
        do i=1,nsource
           if (ifpot .eq. 1) pot2(i)=0
           if (iffld .eq. 1) then
              fld2(1,i)=0
              fld2(2,i)=0
              fld2(3,i)=0
           endif
        enddo
        t1=omp_get_wtime()
C$OMP PARALLEL DO DEFAULT(SHARED)
C$OMP$PRIVATE(i,j,ptemp,ftemp) 
        do j=1,m
           do i=1,nsource       
              if( i .eq. j ) cycle
              if( ifcharge .eq. 1 ) then
                 call lpotfld3d(iffld,source(1,i),charge(i),
     $              source(1,j),ptemp,ftemp)
                 if (ifpot .eq. 1) pot2(j)=pot2(j)+ptemp
                 if (iffld .eq. 1) then
                    fld2(1,j)=fld2(1,j)+ftemp(1)
                    fld2(2,j)=fld2(2,j)+ftemp(2)
                    fld2(3,j)=fld2(3,j)+ftemp(3)
                 endif
              endif
              if (ifdipole .eq. 1) then
                 call lpotfld3d_dp(iffld,source(1,i),
     $              dipstr(i),dipvec(1,i),
     $              source(1,j),ptemp,ftemp)
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
        t2=omp_get_wtime()
        print*,'Direct =',t2-t1
        call l3derror(pot,pot2,m,aerr,rerr)
        print*,'Err pot=',rerr
        call l3derror(fld,fld2,3*m,aerr,rerr)
        print*,'Err acc=',rerr
        stop
        end
c
        subroutine l3derror(pot1,pot2,n,ae,re)
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
