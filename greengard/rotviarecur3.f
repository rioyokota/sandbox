C*****************************************************************
        subroutine rotviarecur3s(nterms,mpole,ld1,marray,
     1                         ld2,rd1,rd2,sqc,theta,ldc)
C*****************************************************************
c	Fast, recursive method for applying rotation matrix about
c	the y-axis determined by angle theta.
c       The rotation matrices for each order (first index) are computed
c       from the lowest to the highest. As each one is generated, it
c       is applied to the input expansion "mpole" and overwritten.
c
c       As a result, it is sufficient to use two arrays rd1 and rd2 for
c       the two term recurrence, rather than storing them for all orders
c       as in the original code. There is some loss in speed
c       if the rotation operator is to be used multiple times, but the 
c       memory savings is often more critical.
c
c       Use symmetry properties of rotation matrices
C---------------------------------------------------------------------
c       INPUT:
c
c       nterms: dimension parameter for d - the rotation matrix.
C       mpole   coefficients of original multiple expansion
C       rd1     work space 
C       rd2     work space
c       sqc:    an array contains the square roots of the
c               binomial coefficients.
c       theta:  the rotate angle about the y-axis.
c       ldc     dimensions of sqc array
c
C---------------------------------------------------------------------
c       OUTPUT:
c
c       marray  coefficients of rotated expansion.
c
C---------------------------------------------------------------------
	implicit none
        integer ld1,ld2,nterms,ldc
        integer ij,im,imp,m,mp
        real *8 theta
        real *8 ww,done,ctheta,stheta,hsthta,cthtap,cthtan,d
        real *8 precis,scale
	real *8 rd1(0:ldc,-ldc:ldc)
	real *8 rd2(0:ldc,-ldc:ldc)
	real *8 sqc(0:2*ldc,2)
	complex *16 mpole(0:ld1,-ld1:ld1)
	complex *16 marray(0:ld2,-ld2:ld2)
	data precis/1.0d-20/
        ww=1/sqrt(2.0d0)
        do m = 0, 2*ldc 
	   sqc(m,1) = dsqrt(m+0.0d0)
        enddo
	sqc(0,2) = 0.0d0
	if( ldc .gt. 0 ) sqc(1,2) = 0.0d0
        do m = 2, 2*ldc 
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
      marray(0,0)=mpole(0,0)*rd1(0,0)
c ... Loop over first index ij=1,nterms, constructing
c     rotation matrices recursively.
      do ij=1,nterms
c     For mprime=0, use formula (1).
         do im=-ij,-1
	    rd2(0,im)=-sqc(ij-im,2)*rd1(0,im+1)
	    if (im.gt.(1-ij)) then
	       rd2(0,im)=rd2(0,im)+sqc(ij+im,2)*rd1(0,im-1)
	    endif
	    rd2(0,im)=rd2(0,im)*hsthta
	    if (im.gt.-ij) then
	       rd2(0,im)=rd2(0,im)+
     1         rd1(0,im)*ctheta*sqc(ij+im,1)*sqc(ij-im,1)
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
c        For 0<mprime<=j (2nd index) case, use formula (2).
	 do imp=1,ij
            scale=(ww/sqc(ij+imp,2))
	    do im=imp,ij
	       rd2(imp,+im)=rd1(imp-1,+im-1)*(cthtap*sqc(ij+im,2))
	       rd2(imp,-im)=rd1(imp-1,-im+1)*(cthtan*sqc(ij+im,2))
	       if (im.lt.(ij-1)) then
	          rd2(imp,+im)=rd2(imp,+im)-rd1(imp-1,+im+1)*
     $               (cthtan*sqc(ij-im,2))
	          rd2(imp,-im)=rd2(imp,-im)-rd1(imp-1,-im-1)*
     $               (cthtap*sqc(ij-im,2))
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
            marray(ij,m)=mpole(ij,0)*rd2(0,m)
            do mp=1,ij
               marray(ij,m)=marray(ij,m)+
     1	       mpole(ij,mp)*rd2(mp,m)+
     1         mpole(ij,-mp)*rd2(mp,-m)
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
C*****************************************************************
        subroutine rotviarecur3f90(theta,nterms,mpole,ld1,
     1     marray,ld2)
C*****************************************************************
c
c       Purpose:
c
c	Fast, recursive method for applying rotation matrix about
c	the y-axis determined by angle theta.
c       The rotation matrices for each order (first index) are computed
c       from the lowest to the highest. As each one is generated, it
c       is applied to the input expansion "mpole" and overwritten.
c
c       As a result, it is sufficient to use two arrays rd1 and rd2 for
c       the two term recurrence, rather than storing them for all orders
c       as in the original code. There is some loss in speed
c       if the rotation operator is to be used multiple times, but the 
c       memory savings is often more critical.
c
c       Note that this is just a driver to rotviarecur3s and
c       rotviarecur3t routines, the main purpose of this routine is to
c       have the calling sequence compatible with rotproj.
c
C---------------------------------------------------------------------
c       INPUT:
c
c       theta:  the rotation angle about the y-axis.
c       nterms: order of multipole expansion
c
c       m1    :  max m index for first expansion  
c       m2    :  max m index for second expansion 
C       mpole :  coefficients of original multiple expansion
C       ld1   :  leading dim for mpole (must exceed nterms)
C       ld2   :  leading dim for marray (must exceed nterms)
c
C---------------------------------------------------------------------
c       OUTPUT:
c
c       marray   coefficients of rotated expansion.
c       lused    amount of workspace used.
c
C---------------------------------------------------------------------
	implicit none
        integer ld1,ld2,nterms,m1,m2
        integer ldc,ird1,lrd,ird2,isqc,lused,ier
        real *8 theta
        real *8, allocatable :: w(:)
	complex *16 mpole(0:ld1,-ld1:ld1)
	complex *16 marray(0:ld2,-ld2:ld2)
        ldc = nterms
        ird1 = 1
        lrd  = (ldc+1)*(2*ldc+1) + 3
        ird2 = ird1 + lrd
        isqc = ird2 + lrd
        lused = isqc + 2*(2*ldc+1) 
        allocate (w(lused), stat=ier)
        if( ier.ne.0 ) return
        call rotviarecur3s(nterms,mpole,ld1,marray,
     1                    ld2,w(ird1),w(ird2),w(isqc),theta,ldc)
        return
        end
C*****************************************************************
        subroutine rotviarecur3p_init(ier,rotmat,ldc,theta)
C*****************************************************************
c
c       Purpose:
c
c       Precompute the coefficients of rotation matrix. 
c
c       Use symmetry properties of rotation matrices. Fortran 90 code.
c
C---------------------------------------------------------------------
c       INPUT:
c
c       theta:  the rotate angle about the y-axis.
c       ldc     dimensions of rotation matrix array
c
C---------------------------------------------------------------------
c       OUTPUT:
c
c       ier: error return code
c       rotmat: coefficients of rotation matrix
c
C---------------------------------------------------------------------
	implicit none
        integer ld1,ld2,nterms,m1,m2,ldc
        integer ier,ij,im,imp,m,mp
        real *8 theta
        real *8 ww,done,ctheta,stheta,hsthta,cthtap,cthtan,d
        real *8 precis,scale
	real *8 rotmat(0:ldc,0:ldc,-ldc:ldc)
	real *8, allocatable :: rd1(:,:)
	real *8, allocatable :: rd2(:,:)
	real *8, allocatable :: sqc(:,:)
	data precis/1.0d-20/
	allocate (rd1(0:ldc,-ldc:ldc),stat=ier)
        if(ier .ne. 0 ) return
	allocate (rd2(0:ldc,-ldc:ldc),stat=ier)
        if(ier .ne. 0 ) return
	allocate (sqc(0:2*ldc,2),stat=ier)
        if(ier .ne. 0 ) return
        ww=1/sqrt(2.0d0)
        do m = 0, 2*ldc 
	   sqc(m,1) = dsqrt(m+0.0d0)
        enddo
	sqc(0,2) = 0.0d0
	if( ldc .gt. 0 ) sqc(1,2) = 0.0d0
        do m = 2, 2*ldc 
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
      rotmat(0,0,0)=rd1(0,0)
c ... Loop over first index ij=1,nterms, constructing
c     rotation matrices recursively.
      nterms=ldc
      do ij=1,nterms
c     For mprime=0, use formula (1).
         do im=-ij,-1
	    rd2(0,im)=-sqc(ij-im,2)*rd1(0,im+1)
	    if (im.gt.(1-ij)) then
	       rd2(0,im)=rd2(0,im)+sqc(ij+im,2)*rd1(0,im-1)
	    endif
	    rd2(0,im)=rd2(0,im)*hsthta
	    if (im.gt.-ij) then
	       rd2(0,im)=rd2(0,im)+
     1         rd1(0,im)*ctheta*sqc(ij+im,1)*sqc(ij-im,1)
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
c        For 0<mprime<=j (2nd index) case, use formula (2).
	 do imp=1,ij
            scale=(ww/sqc(ij+imp,2))
	    do im=imp,ij
	       rd2(imp,+im)=rd1(imp-1,+im-1)*(cthtap*sqc(ij+im,2))
	       rd2(imp,-im)=rd1(imp-1,-im+1)*(cthtan*sqc(ij+im,2))
	       if (im.lt.(ij-1)) then
	          rd2(imp,+im)=rd2(imp,+im)-rd1(imp-1,+im+1)*
     $               (cthtan*sqc(ij-im,2))
	          rd2(imp,-im)=rd2(imp,-im)-rd1(imp-1,-im-1)*
     $               (cthtap*sqc(ij-im,2))
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
            do mp=0,ij
	       rd1(mp,m) = rd2(mp,m)
               rotmat(mp,ij,m) = rd2(mp,m)
            enddo
         enddo
      enddo
      return
      end
C*****************************************************************
        subroutine rotviarecur3p_apply
     $     (theta,nterms,m1,m2,mpole,ld1,marray,ld2,rotmat,ldc)
C*****************************************************************
c
c       Purpose:
c
c	Fast, recursive method for applying rotation matrix about
c	the y-axis determined by angle theta.
c
C---------------------------------------------------------------------
c       INPUT:
c
c       theta:  the rotation angle about the y-axis.
c       nterms: order of multipole expansion
c
c       m1    :  max m index for first expansion  
c       m2    :  max m index for second expansion 
C       mpole :  coefficients of original multiple expansion
C       ld1   :  leading dim for mpole (must exceed nterms)
C       ld2   :  leading dim for marray (must exceed nterms)
c       rotmat:  real *8 (0:ldc,0:ldc,-ldc:ldc): rotation matrix 
c       ldc   :  leading dim for rotation matrix (must exceed nterms)
c
C---------------------------------------------------------------------
c       OUTPUT:
c
c       marray   coefficients of rotated expansion.
c
C---------------------------------------------------------------------
	implicit none
	integer nterms,m1,m2,ld1,ld2,ldc
	integer ij,m,mp
	real *8 theta
	complex *16 mpole(0:ld1,-ld1:ld1)
	complex *16 marray(0:ld2,-ld2:ld2)
	real *8 rotmat(0:ldc,0:ldc,-ldc:ldc)
        if (m1.lt.nterms .or. m2.lt.nterms) then
c       ... truncate multipole expansions
           do ij=0,nterms

              do m=-ij,ij
                  marray(ij,m)=0
              enddo
              do m=-min(ij,m2),min(ij,m2)
                 marray(ij,m)=mpole(ij,0)*rotmat(0,ij,m) 
                 do mp=1,min(ij,m1)
                    marray(ij,m)=marray(ij,m)+
     1	            mpole(ij,mp)*rotmat(mp,ij,m)+
     1              mpole(ij,-mp)*rotmat(mp,ij,-m)
                 enddo
              enddo
           enddo
        else
c       ... apply rotation matrix
           do ij=0,nterms
              do m=-ij,ij
                 marray(ij,m)=mpole(ij,0)*rotmat(0,ij,m) 
                 do mp=1,ij
                    marray(ij,m)=marray(ij,m)+
     1	            mpole(ij,mp)*rotmat(mp,ij,m)+
     1              mpole(ij,-mp)*rotmat(mp,ij,-m)
                 enddo
              enddo
           enddo
        endif
        return
        end
