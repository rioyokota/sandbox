c      1)  Hankel and Bessel functions are consistently scaled as
c       	hvec(n)= h_n(z)*scale^(n)
c       	jvec(n)= j_n(z)/scale^(n)
c
c          In some earlier FMM implementations, the convention
c       	hvec(n)= h_n(z)*scale^(n+1)
c          was sometimes used, leading to various obscure rescaling 
c          steps.
c
c          scale should be of the order of |z| if |z| < 1. Otherwise,
c          scale should be set to 1.
c
c
c      2) There are many definitions of the spherical harmonics,
c         which differ in terms of normalization constants. We
c         adopt the following convention:
c
c         For m>0, we define Y_n^m according to 
c
c         Y_n^m = \sqrt{2n+1} \sqrt{\frac{ (n-m)!}{(n+m)!}} \cdot
c                 P_n^m(\cos \theta)  e^{i m phi} 
c         and
c 
c         Y_n^-m = dconjg( Y_n^m )
c    
c         We omit the Condon-Shortley phase factor (-1)^m in the 
c         definition of Y_n^m for m<0. (This is standard in several
c         communities.)
c
c         We also omit the factor \sqrt{\frac{1}{4 \pi}}, so that
c         the Y_n^m are orthogonal on the unit sphere but not 
c         orthonormal.  (This is also standard in several communities.)
c         More precisely, 
c
c                 \int_S Y_n^m Y_n^m d\Omega = 4 \pi. 
c
c         Using our standard definition, the addition theorem takes 
c         the simple form 
c
c         e^( i k r}/(ikr) = 
c         \sum_n \sum_m  j_n(k|S|) Ylm*(S) h_n(k|T|) Ylm(T)
c
c
c**********************************************************************
      subroutine cart2polar(zat,r,theta,phi)
c**********************************************************************
c     Convert from Cartesian to polar coordinates.
c-----------------------------------------------------------------------
c     INPUT:
c	zat   :  Cartesian vector
c-----------------------------------------------------------------------
c     OUTPUT:
c	r     :  |zat|
c	theta : angle subtended with respect to z-axis
c	phi   : angle of (zat(1),zat(2)) subtended with 
c               respect to x-axis
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 zat(3)
      complex *16 ephi,eye
      data eye/(0.0d0,1.0d0)/
      r= sqrt(zat(1)**2+zat(2)**2+zat(3)**2)
      proj = sqrt(zat(1)**2+zat(2)**2)
      theta = datan2(proj,zat(3))
      if( abs(zat(1)) .eq. 0 .and. abs(zat(2)) .eq. 0 ) then
      phi = 0
      else
      phi = datan2(zat(2),zat(1))
      endif
      return
      end
c**********************************************************************
      subroutine h3d01(z,h0,h1)
c**********************************************************************
c     Compute spherical Hankel functions of order 0 and 1 
c     h0(z)  =   exp(i*z)/(i*z),
c     h1(z)  =   - h0' = -h0*(i-1/z) = h0*(1/z-i)
c-----------------------------------------------------------------------
c     INPUT:
c	z   :  argument of Hankel functions
c              if abs(z)<1.0d-15, returns zero.
c-----------------------------------------------------------------------
c     OUTPUT:
c	h0  :  h0(z)    (spherical Hankel function of order 0).
c	h1  :  -h0'(z)  (spherical Hankel function of order 1).
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      complex *16 z,zinv,eye,cd,h0,h1
      data eye/(0.0d0,1.0d0)/, thresh/1.0d-15/, done/1.0d0/
      if (abs(z).lt.thresh) then
         h0=0.0d0
         h1=0.0d0
         return
      endif
      cd = eye*z
      h0=exp(cd)/cd
      h1=h0*(done/z - eye)
      return
      end
c**********************************************************************
      subroutine h3dall(nterms,z,scale,hvec,ifder,hder)
c**********************************************************************
c     This subroutine computes scaled versions of the spherical Hankel 
c     functions h_n of orders 0 to nterms.
c
c       	hvec(n)= h_n(z)*scale^(n)
c
c     The parameter SCALE is useful when |z| < 1, in which case
c     it damps out the rapid growth of h_n as n increases. In such 
c     cases, we recommend setting 
c                                 
c               scale = |z|
c
c     or something close. If |z| > 1, set scale = 1.
c
c     If the flag IFDER is set to one, it also computes the 
c     derivatives of h_n.
c
c		hder(n)= h_n'(z)*scale^(n)
c
c     NOTE: If |z| < 1.0d-15, the subroutine returns zero.
c-----------------------------------------------------------------------
c     INPUT:
c     nterms  : highest order of the Hankel functions to be computed.
c     z       : argument of the Hankel functions.
c     scale   : scaling parameter discussed above
c     ifder   : flag indcating whether derivatives should be computed.
c		ifder = 1   ==> compute 
c		ifder = 0   ==> do not compute
c-----------------------------------------------------------------------
c     OUTPUT:
c     hvec    : the vector of spherical Hankel functions 
c     hder    : the derivatives of the spherical Hankel functions 
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      complex *16 hvec(0:nterms),hder(0:nterms)
      complex *16 zk2,z,zinv,ztmp,fhextra
      data thresh/1.0d-15/,done/1.0d0/
c     If |z| < thresh, return zeros.
      if (abs(z).lt.thresh) then
         do i=0,nterms
            hvec(i)=0
            hder(i)=0
         enddo
         return
      endif
c     Otherwise, get h_0 and h_1 analytically and the rest via recursion.
      call h3d01(z,hvec(0),hvec(1))
      hvec(0)=hvec(0)
      hvec(1)=hvec(1)*scale
c     From Abramowitz and Stegun (10.1.19)
c     h_{n+1}(z)=(2n+1)/z * h_n(z) - h_{n-1}(z)
c     With scaling:
c     hvec(n+1)=scale*(2n+1)/z * hvec(n) -(scale**2) hvec(n-1)
      scal2=scale*scale
      zinv=scale/z
      do i=1,nterms-1
	 dtmp=(2*i+done)
	 ztmp=zinv*dtmp
	 hvec(i+1)=ztmp*hvec(i)-scal2*hvec(i-1)
      enddo
c     From Abramowitz and Stegun (10.1.21)
c     h_{n}'(z)= h_{n-1}(z) - (n+1)/z * h_n(z)
c     With scaling:
c     hder(n)=scale* hvec(n-1) - (n+1)/z * hvec(n)
      if (ifder.eq.1) then
	 hder(0)=-hvec(1)/scale
         zinv=1.0d0/z
         do i=1,nterms
	    dtmp=(i+done)
	    ztmp=zinv*dtmp
	    hder(i)=scale*hvec(i-1)-ztmp*hvec(i)
	 enddo
      endif
      return
      end

c**********************************************************************
      subroutine P2P(ibox,target,pot,fld,jbox,source,charge,zk) 
c**********************************************************************
c     This subroutine calculates the potential POT and field FLD
c     at the target point TARGET, due to a charge at 
c     SOURCE. The scaling is that required of the delta function
c     response: i.e.,
c     
c              	pot = exp(i*k*r)/r
c		fld = -grad(pot)
c---------------------------------------------------------------------
c     INPUT:
c     source    : location of the source 
c     charge    : charge strength
c     target    : location of the target
c     zk        : helmholtz parameter
c---------------------------------------------------------------------
c     OUTPUT:
c     pot       : calculated potential
c     fld       : calculated gradient
c---------------------------------------------------------------------
      implicit none
      integer i,j,ibox(20),jbox(20)
      real *8 dx,dy,dz,R2,R,target(3,1000000),source(3,1000000)
      complex *16 i1,zk,coef1,coef2
      complex *16 charge(1000000),pot(1000000),fld(3,1000000)
      data i1/(0.0d0,1.0d0)/
      do i=ibox(14),ibox(14)+ibox(15)-1
         do j=jbox(14),jbox(14)+jbox(15)-1
            dx=target(1,i)-source(1,j)
            dy=target(2,i)-source(2,j)
            dz=target(3,i)-source(3,j)
            R2=dx*dx+dy*dy+dz*dz
            if(R2.eq.0) cycle
            R=sqrt(R2)
            coef1=charge(j)*cdexp(i1*zk*R)/R
            coef2=(1-i1*zk*R)*coef1/R2
            pot(i)=pot(i)+coef1
            fld(1,i)=fld(1,i)+coef2*dx
            fld(2,i)=fld(2,i)+coef2*dy
            fld(3,i)=fld(3,i)+coef2*dz
         enddo
      enddo
      return
      end
c**********************************************************************
      subroutine L2P(wavek,rscale,center,locexp,nterms,
     1     nterms1,lwfjs,target,nt,pot,fld,wlege,nlege,ier)
c**********************************************************************
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point TARGET. 
c
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c---------------------------------------------------------------------
c     INPUT:
c     wavek      : the Helmholtz coefficient
c     rscale     : scaling parameter used in forming expansion
c     center     : coordinates of the expansion center
c     locexp     : coeffs of the j-expansion
c     nterms     : order of the h-expansion
c     nterms1    : order of the truncated expansion
c     target(3)   : target vector
c     nt         : number of targets
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
c---------------------------------------------------------------------
c     OUTPUT:
c     ier        : error return code
c     pot        : potential at target (if requested)
c     fld(3)     : gradient at target (if requested)
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c---------------------------------------------------------------------
      implicit none
      integer i,j,m,n,nt,ier,jer,nterms,nterms1,nlege,ntop,lwfjs
      integer iscale(0:lwfjs)
      real *8 r,rx,ry,rz,theta,thetax,thetay,thetaz,rscale
      real *8 phi,phix,phiy,phiz,ctheta,stheta,cphi,sphi,wlege
      real *8 center(3),target(3,1),dX(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 wavek,pot(1),fld(3,1),ephi1,ephi1inv
      complex *16 locexp(0:nterms,-nterms:nterms)
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjsuse,fjs(0:lwfjs),fjder(0:lwfjs)
      complex *16 eye,ur,utheta,uphi
      complex *16 ztmp,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
      data eye/(0.0d0,1.0d0)/
      ier=0
      do i=1,nt
         dX(1)=target(1,i)-center(1)
         dX(2)=target(2,i)-center(2)
         dX(3)=target(3,i)-center(3)
         call cart2polar(dX,r,theta,phi)
         ctheta = dcos(theta)
         stheta=sqrt(1-ctheta*ctheta)
         cphi = dcos(phi)
         sphi = dsin(phi)
         ephi1 = dcmplx(cphi,sphi)
         ephi(0)=1.0d0
         ephi(1)=ephi1
         ephi(-1)=dconjg(ephi1)
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi1
            ephi(-j)=ephi(-j+1)*ephi(-1)
         enddo
         rx = stheta*cphi
         thetax = ctheta*cphi
         phix = -sphi
         ry = stheta*sphi
         thetay = ctheta*sphi
         phiy = cphi
         rz = ctheta
         thetaz = -stheta
         phiz = 0.0d0
         call ylgndr2sfw(nterms1,ctheta,pp,ppd,wlege,nlege)
         z=wavek*r
         call jfuns3d(jer,nterms1,z,rscale,fjs,1,fjder,
     1	      lwfjs,iscale,ntop)
         if (jer.ne.0) then
            ier=8
            return
         endif
         pot(i)=pot(i)+locexp(0,0)*fjs(0)
         do j=0,nterms1
            fjder(j)=fjder(j)*wavek
         enddo
         ur = locexp(0,0)*fjder(0)
         utheta = 0.0d0
         uphi = 0.0d0
         do n=1,nterms1
            pot(i)=pot(i)+locexp(n,0)*fjs(n)*pp(n,0)
            ur = ur + fjder(n)*pp(n,0)*locexp(n,0)
            fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
            fjsuse = wavek*fjsuse/(2*n+1.0d0)
            utheta = utheta -locexp(n,0)*fjsuse*ppd(n,0)*stheta
            do m=1,n
               ztmp1=fjs(n)*pp(n,m)*stheta
               ztmp2 = locexp(n,m)*ephi(m) 
               ztmp3 = locexp(n,-m)*ephi(-m)
               ztmpsum = ztmp2+ztmp3
               pot(i)=pot(i)+ztmp1*ztmpsum
               ur = ur + fjder(n)*pp(n,m)*stheta*ztmpsum
               utheta = utheta -ztmpsum*fjsuse*ppd(n,m)
               ztmpsum = eye*m*(ztmp2 - ztmp3)
               uphi = uphi + fjsuse*pp(n,m)*ztmpsum
            enddo
         enddo
         ux = ur*rx + utheta*thetax + uphi*phix
         uy = ur*ry + utheta*thetay + uphi*phiy
         uz = ur*rz + utheta*thetaz + uphi*phiz
         fld(1,i) = fld(1,i)-ux
         fld(2,i) = fld(2,i)-uy
         fld(3,i) = fld(3,i)-uz
      enddo
      return
      end

C***********************************************************************
      subroutine P2M
     1     (ier,zk,rscale,source,charge,ns,center,
     1     nterms,nterms1,lwfjs,mpole,wlege,nlege)
C***********************************************************************
C
C     Constructs multipole (h) expansion about CENTER due to NS sources 
C     located at SOURCES(3,*).
C
c-----------------------------------------------------------------------
C     INPUT:
c
C     zk              : Helmholtz parameter 
C     scale           : the scaling factor.
C     sources         : coordinates of sources
C     charge          : source strengths
C     ns              : number of sources
C     center          : epxansion center
C     nterms          : order of multipole expansion
C     nterms1         : order of truncated expansion
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
C
c-----------------------------------------------------------------------
C     OUTPUT:
C
c     ier             : error return code
c     mpole           : coeffs of the h-expansion
c-----------------------------------------------------------------------
      implicit none
      integer i,j,m,l,n,ns,nterms,nterms1,nlege,ifder,lwfjs,jer,ntop,ier
      integer iscale(0:lwfjs)
      real *8 r,theta,phi,ctheta,stheta,cphi,sphi,wlege,rscale,dtmp
      real *8 thresh
      real *8 center(3),source(3,ns),dX(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 charge(ns),i1,zk,z,ztmp,ephi1,ephi1inv
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjs(0:lwfjs),fjder(0:lwfjs)
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 mtemp(0:nterms,-nterms:nterms)
      data i1/(0.0d0,1.0d0)/
      data thresh/1.0d-15/
      ier=0
      do l = 0,nterms
         do m=-l,l
            mtemp(l,m) = 0
         enddo
      enddo
      do i = 1, ns
         dX(1)=source(1,i)-center(1)
         dX(2)=source(2,i)-center(2)
         dX(3)=source(3,i)-center(3)
         call cart2polar(dX,r,theta,phi)
         ctheta=dcos(theta)
         stheta=dsin(theta)
         cphi=dcos(phi)
         sphi=dsin(phi)
         ephi1=dcmplx(cphi,sphi)
         ephi(0)=1.0d0
         ephi(1)=ephi1
         ephi(-1)=dconjg(ephi1)
         do j=2,nterms+1
            ephi(j)=ephi(j-1)*ephi1
            ephi(-j)=ephi(-j+1)*ephi(-1)
         enddo
         call ylgndrfw(nterms1,ctheta,pp,wlege,nlege)
         ifder=0
         z=zk*r
         call jfuns3d(jer,nterms1,z,rscale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
         do n = 0,nterms1
            fjs(n) = fjs(n)*charge(i)
         enddo
         mtemp(0,0)= mtemp(0,0) + fjs(0)
         do n=1,nterms1
            dtmp=pp(n,0)
            mtemp(n,0)= mtemp(n,0) + dtmp*fjs(n)
            do m=1,n
               ztmp=pp(n,m)*fjs(n)
               mtemp(n, m)= mtemp(n, m) + ztmp*dconjg(ephi(m))
               mtemp(n,-m)= mtemp(n,-m) + ztmp*dconjg(ephi(-m))
            enddo
         enddo
      enddo
      do l = 0,nterms
         do m=-l,l
            mpole(l,m) = mpole(l,m)+mtemp(l,m)*i1*zk
         enddo
      enddo
      return
      end
