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
c-----------------------------------------------------------------------
c
c      h3dformmp: creates multipole expansion (outgoing) due to 
c                 a collection of charges.
c
c      cart2polar: utility function.
c                  converts Cartesian coordinates into polar
c                  representation needed by other routines.
c
c      h3d01: computes h0, h1 (first two spherical Hankel fns.)
c      h3dall: computes Hankel functions of all orders and scales them
c
c      h3dtaeval: computes potential and -grad(potential) 
c                 due to local expansion
c                 at a single target.
c
c      h3dformta: creates local expansion due to 
c                 a collection of charges.
c
c      hpotfld3dall:  direct calculation for a collection of charge sources
c      hpotfld3d : direct calculation for a single charge source
c
c      h3dtaevalall_trunc: computes potential and -grad(potential)
c                 due to a local expansion (OPTIMIZED VERSION)
c                 at a collection of targets.
c
c      h3dtaeval_trunc: computes potential and -grad(potential)
c                 due to a local expansion (OPTIMIZED VERSION)
c                 at a single target.
c
c      h3dformmp_trunc: creates multipole expansion (outgoing) 
c                 due to a collection of charges (OPTIMIZED VERSION).
c
c      h3dformmp_add_trunc: *increments* multipole expansion (outgoing) 
c                 due to a collection of charges (OPTIMIZED VERSION).
c
c      h3dformta_trunc: creates local expansion (incoming) due to 
c                 a collection of charges (OPTIMIZED VERSION).
c
c      h3dformta_add_trunc: *increments* local expansion (incoming) due to 
c                 a collection of charges (OPTIMIZED VERSION).
c
c      h3dformta_trunc: creates local expansion (incoming) due to 
c                 a collection of dipoles (OPTIMIZED VERSION).
c
c      h3dformta_add_trunc: *increments* local expansion (incoming) due to 
c                 a collection of dipoles (OPTIMIZED VERSION).
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
      complex *16 hvec(0:1),hder(0:1)
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
      subroutine h3dtaeval(wavek,rscale,center,locexp,nterms,
     1		ztarg,pot,fld,ier)
c**********************************************************************
c
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point ZTARG. 
c
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c
c---------------------------------------------------------------------
c     INPUT:
c
c     wavek      : the Helmholtz coefficient
c     rscale     : scaling parameter used in forming expansion
c                                   (see h3dformmp1)
c     center     : coordinates of the expansion center
c     locexp     : coeffs of the j-expansion
c     nterms     : order of the h-expansion
c     ztarg(3)   : target vector
c---------------------------------------------------------------------
c     OUTPUT:
c
c     ier        : error return code
c		      ier=0	returned successfully
c		      ier=8 insuffficient workspace 
c		      ier=16 insufficient memory 
c                            in subroutine "jfuns3d"
c     pot        : potential at ztarg(3)
c     fld(3)     : gradient at ztarg (if requested)
c     lused      : amount of work space "w" used
c
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer lwfjs
      real *8 center(3),ztarg(3)
      real *8, allocatable :: w(:)
      complex *16 wavek,pot,fld(3)
      complex *16 locexp(0:nterms,-nterms:nterms)
c
c ... Assigning work spaces for various temporary arrays:
c
      ier=0
c
      lwfjs=nterms+1000
      ipp=1
      lpp=(nterms+1)**2+3
      ippd  = ipp+lpp
c
      iephi=ippd+lpp
      lephi=2*(2*nterms+1)+7
c
      iiscale=iephi+lephi
      liscale=(lwfjs+1)+3
c
      ifjs=iiscale+liscale
      lfjs=2*(lwfjs+1)+3
c
      ifjder=ifjs+lfjs
      lfjder=2*(nterms+1)+3
c
      lused=ifjder+lfjder
      allocate(w(lused))
c
      call h3dtaeval0(jer,wavek,rscale,center,locexp,nterms,ztarg,
     1	     pot,fld,w(ipp),w(ippd),w(iephi),w(ifjs),
     2       w(ifjder),lwfjs,w(iiscale))
      if (jer.ne.0) ier=16
c
      return
      end
c
c
c
c**********************************************************************
      subroutine h3dtaeval0(ier,wavek,rscale,center,locexp,nterms,
     1		ztarg,pot,fld,pp,ppd,ephi,fjs,fjder,lwfjs,iscale)
c**********************************************************************
c
c     See h3dtaeval for comments.
c     (pp and ppd are storage arrays for Ynm and Ynm')
c
c----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer iscale(0:1)
      real *8 center(3),ztarg(3),zdiff(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 wavek,pot,fld(3),ephi1,ephi1inv
      complex *16 locexp(0:nterms,-nterms:nterms)
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjsuse,fjs(0:1),fjder(0:1)
c
      complex *16 eye,ur,utheta,uphi
      complex *16 ztmp,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
c
      data eye/(0.0d0,1.0d0)/
c
      ier=0
      done=1.0d0
c
      zdiff(1)=ztarg(1)-center(1)
      zdiff(2)=ztarg(2)-center(2)
      zdiff(3)=ztarg(3)-center(3)
c
c     Convert to spherical coordinates
c
      call cart2polar(zdiff,r,theta,phi)
      ctheta = dcos(theta)
      stheta=sqrt(done-ctheta*ctheta)
      cphi = dcos(phi)
      sphi = dsin(phi)
      ephi1 = dcmplx(cphi,sphi)
c
c     compute e^{eye*m*phi} array.
c
c
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=dconjg(ephi1)
      do i=2,nterms+1
         ephi(i)=ephi(i-1)*ephi1
         ephi(-i)=ephi(-i+1)*ephi(-1)
      enddo
c
c     compute coefficients in change of variables from spherical
c     to Cartesian gradients. In phix, phiy, we leave out the 
c     1/sin(theta) contribution, since we use values of Ynm (which
c     multiplies phix and phiy) that are scaled by 
c     1/sin(theta).
c
c     In thetax, thetaty, phix, phiy we leave out the 1/r factors in the 
c     change of variables to avoid blow-up at the origin.
c     For the n=0 mode, it is not relevant. For n>0 modes,
c     we use the recurrence relation 
c
c     (2n+1)fjs_n(kr)/(kr) = fjs(n+1)*rscale + fjs(n-1)/rscale
c
c     to avoid division by r. The variable fjsuse is set to fjs(n)/r:
c
c           fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
c	    fjsuse = wavek*fjsuse/(2*n+1.0d0)
c
c     
c
      rx = stheta*cphi
      thetax = ctheta*cphi
      phix = -sphi
      ry = stheta*sphi
      thetay = ctheta*sphi
      phiy = cphi
      rz = ctheta
      thetaz = -stheta
      phiz = 0.0d0
c
c     get the associated Legendre functions:
c
      call ylgndr2s(nterms,ctheta,pp,ppd)
c
c     get the spherical Bessel functions and their derivatives.
c
      ifder=1
      z=wavek*r
      call jfuns3d(jer,nterms,z,rscale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
      if (jer.ne.0) then
         ier=8
         return
      endif
c
c     scale derivatives of Bessel functions so that they are
c     derivatives with respect to r.
c
c
      pot=locexp(0,0)*fjs(0)
      do i=0,nterms
         fjder(i)=fjder(i)*wavek
      enddo
      ur = locexp(0,0)*fjder(0)
      utheta = 0.0d0
      uphi = 0.0d0
c
c     compute the potential and the field:
c
      do n=1,nterms
         pot=pot+locexp(n,0)*fjs(n)*pp(n,0)
         ur = ur + fjder(n)*pp(n,0)*locexp(n,0)
         fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
         fjsuse = wavek*fjsuse/(2*n+1.0d0)
         utheta = utheta -locexp(n,0)*fjsuse*ppd(n,0)*stheta
         do m=1,n
            ztmp1=fjs(n)*pp(n,m)*stheta
            ztmp2 = locexp(n,m)*ephi(m) 
            ztmp3 = locexp(n,-m)*ephi(-m)
            ztmpsum = ztmp2+ztmp3
            pot=pot+ztmp1*ztmpsum
            ur = ur + fjder(n)*pp(n,m)*stheta*ztmpsum
            utheta = utheta -ztmpsum*fjsuse*ppd(n,m)
            ztmpsum = eye*m*(ztmp2 - ztmp3)
            uphi = uphi + fjsuse*pp(n,m)*ztmpsum
         enddo
      enddo
      ux = ur*rx + utheta*thetax + uphi*phix
      uy = ur*ry + utheta*thetay + uphi*phiy
      uz = ur*rz + utheta*thetaz + uphi*phiz
      fld(1) = -ux
      fld(2) = -uy
      fld(3) = -uz
      return
      end
c
c
c
c
c
c**********************************************************************
      subroutine h3dformta(ier,zk,rscale,sources,charge,ns,center,
     1		           nterms,locexp)
c**********************************************************************
c
c     This subroutine creates a local (j) expansion about the point
c     CENTER due to the NS sources at the locations SOURCES(3,*).
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformta1/h3dformta0 below.
c
c----------------------------------------------------------------------
c     INPUT:
c
c     zk       : Helmholtz coefficient
c     rscale   : scaling parameter
c                     should be less than one in magnitude.
c                     Needed for low frequency regime only
c                     with rsclale abs(wavek) recommended.
c     sources   : coordinates of the sources
c     charge    : charge strengths
c     ns        : number of sources
c     center    : coordinates of the expansion center
c     nterms    : order of the j-expansion
c----------------------------------------------------------------------
c     OUTPUT:
c
c     ier       : error return code
c		  ier=0	returned successfully;
c		  ier=2	insufficient memory in workspace w
c	 	  ier=4  d is out of range in h3dall
c
c     locexp    : coeffs for the j-expansion
c----------------------------------------------------------------------
cf2py intent(in) zk,rscale,sources,charge,ns,center,nterms
cf2py intent(out) ier,locexp
      implicit real *8 (a-h,o-z)
      integer ns
      real *8 sources(3,ns),center(3)
      complex *16 zk,locexp(0:nterms,-nterms:nterms), charge(ns)
      complex *16 eye
      data eye/(0.0d0,1.0d0)/
c
c     initialize local exp
c
      do l = 0,nterms
         do m = -l,l
            locexp(l,m) = 0.0d0
         enddo
      enddo
c
      do i = 1,ns
         call h3dformta1(ier,zk,rscale,sources(1,i),charge(i),
     1		center,nterms,locexp)
      enddo
c
c     scale by (i*k)
c
      do l = 0,nterms
         do m=-l,l
            locexp(l,m) = locexp(l,m)*eye*zk
         enddo
      enddo
C
      return
      end
c
c
c
c
c
c
c**********************************************************************
      subroutine h3dformta1(ier,wavek,rscale,source,charge,center,
     &		nterms,locexp)
c**********************************************************************
c
c     This subroutine creates the local expansion about CENTER
c     due to a single charge located at SOURCE.
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformta0 below.
c
c---------------------------------------------------------------------
c INPUT:
c
c     wavek     : the Helmholtz coefficient
c     rscale    : scaling parameter
c                         should be less than one in magnitude.
c                         Needed for low frequency regime only
c                         with rsclale abs(wavek) recommended.
c     source    : coordinates of the source
c     charge    : coordinates of the source
c     center    : coordinates of the expansion center
c     nterms    : order of the j-expansion
c---------------------------------------------------------------------
c OUTPUT:
c
c     ier    : error return code
c	           ier=0 successful execution
c		   ier=2 insufficient memory in workspace w
c	 	   ier=4 d is out of range in h3dall
c     locexp : coefficients of the local expansion
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 source(3),center(3)
      real *8, allocatable :: w(:)
      complex *16 wavek,locexp(0:nterms,-nterms:nterms), charge
c
c     Carve up workspace
c
      ier=0
c
      ipp=1
      lpp=(nterms+1)**2+7
c
      iephi=ipp+lpp
      lephi=2*(2*nterms+1)+7
c
      ifhs=iephi+lephi
      lfhs=2*(nterms+1)+7
c
      lused=ifhs+lfhs
      allocate(w(lused))
c
      call h3dformta0(jer,wavek,rscale,source,charge,center,
     &		nterms,locexp,w(ipp),w(iephi),w(ifhs))
      if (jer.ne.0) ier=4
c
      return
      end
c
c
c
c**********************************************************************
      subroutine h3dformta0(ier,wavek,rscale,source,charge,
     &		center,nterms,locexp,pp,ephi,fhs)
c**********************************************************************
c
c     See h3dformta/h3dformta1 for comments
c
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 source(3),center(3),zdiff(3)
      real *8 pp(0:nterms,0:nterms)
      complex *16 wavek,locexp(0:nterms,-nterms:nterms), charge
      complex *16 ephi(-nterms:nterms),ephi1,ephi1inv
      complex *16 fhs(0:nterms),ztmp,fhder(0:1),z
      data thresh/1.0d-15/
c
      ier=0
c
      zdiff(1)=source(1)-center(1)
      zdiff(2)=source(2)-center(2)
      zdiff(3)=source(3)-center(3)
c
      done=1
      call cart2polar(zdiff,r,theta,phi)
      ctheta = dcos(theta)
      stheta=sqrt(done-ctheta*ctheta)
      cphi = dcos(phi)
      sphi = dsin(phi)
      ephi1 = dcmplx(cphi,sphi)
c
c     Compute the e^{eye*m*phi} array
c
      ephi1inv=1.0d0/ephi1
c
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=ephi1inv
      do i=2,nterms
         ephi(i)=ephi(i-1)*ephi1
         ephi(-i)=ephi(-i+1)*ephi1inv
      enddo
c
c     get the Ynm
c
      call ylgndr(nterms,ctheta,pp)
c
c     compute Hankel functions and scale them by charge strength.
c
      ifder=0
      z=wavek*r
      if (abs(z).lt.thresh) then
         ier = 4
         return
      endif
      call h3dall(nterms,z,rscale,fhs,ifder,fhder)
      do n = 0, nterms
         fhs(n) = fhs(n)*charge
      enddo
c
c     Compute contributions to locexp
c
      locexp(0,0)=locexp(0,0) + fhs(0)
      do n=1,nterms
         locexp(n,0)=locexp(n,0) + pp(n,0)*fhs(n)
         do m=1,n
            ztmp=pp(n,m)*fhs(n)
	    locexp(n,m)=locexp(n,m) + ztmp*ephi(-m)
	    locexp(n,-m)=locexp(n,-m) + ztmp*ephi(m)
         enddo
      enddo
      return
      end
c
c
c
c**********************************************************************
      subroutine hpotfld3dall(sources,charge,ns,
     1                   target,wavek,pot,fld)
c**********************************************************************
c
c     This subroutine calculates the potential POT and field FLD
c     at the target point TARGET, due to a collection of charges at 
c     SOURCE(3,ns). The scaling is that required of the delta function
c     response: i.e.,
c     
c              	pot = exp(i*k*r)/r
c		fld = -grad(pot)
c
c---------------------------------------------------------------------
c     INPUT:
c
c     sources(3,*)  : location of the sources
c     charge        : charge strengths
c     ns            : number of sources
c     target        : location of the target
c     wavek         : helmholtz parameter
c
c---------------------------------------------------------------------
c     OUTPUT:
c
c     pot   (real *8)        : calculated potential
c     fld   (real *8)        : calculated gradient
c
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 sources(3,ns),target(3)
      complex *16 wavek,pot,fld(3),potloc,fldloc(3)
      complex *16 h0,h1,cd,eye,z,ewavek
      complex *16 charge(ns)
      data eye/(0.0d0,1.0d0)/
      pot = 0.0d0
      fld(1) = 0.0d0
      fld(2) = 0.0d0
      fld(3) = 0.0d0
      do i = 1,ns
         call hpotfld3d(sources(1,i),charge(i),target,wavek,
     1        potloc,fldloc)
         pot = pot + potloc
         fld(1) = fld(1) + fldloc(1)
         fld(2) = fld(2) + fldloc(2)
         fld(3) = fld(3) + fldloc(3)
      enddo
      return
      end
c
c
c
c
c**********************************************************************
      subroutine hpotfld3d(source,charge,target,wavek,pot,fld)
      implicit real *8 (a-h,o-z)
c**********************************************************************
c
c     This subroutine calculates the potential POT and field FLD
c     at the target point TARGET, due to a charge at 
c     SOURCE. The scaling is that required of the delta function
c     response: i.e.,
c     
c              	pot = exp(i*k*r)/r
c		fld = -grad(pot)
c
c---------------------------------------------------------------------
c     INPUT:
c
c     source    : location of the source 
c     charge    : charge strength
c     target    : location of the target
c     wavek     : helmholtz parameter
c
c---------------------------------------------------------------------
c     OUTPUT:
c
c     pot       : calculated potential
c     fld       : calculated gradient
c
c---------------------------------------------------------------------
      real *8 source(3),target(3)
      complex *16 wavek,pot,fld(3)
      complex *16 h0,h1,cd,eye,z,ewavek
      complex *16 charge
      data eye/(0.0d0,1.0d0)/
      xdiff=target(1)-source(1)
      ydiff=target(2)-source(2)
      zdiff=target(3)-source(3)
      dd=xdiff*xdiff+ydiff*ydiff+zdiff*zdiff
      d=sqrt(dd)
      pot=charge*cdexp(eye*wavek*d)/d
      cd=(1-eye*wavek*d)*pot/dd
      fld(1)=cd*xdiff
      fld(2)=cd*ydiff
      fld(3)=cd*zdiff
      return
      end
c
c**********************************************************************
      subroutine h3dtaevalall_trunc(wavek,rscale,center,locexp,nterms,
     1     nterms1,ztarg,nt,pot,fld,wlege,nlege,ier)
c**********************************************************************
c
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point ZTARG. 
c
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c
c---------------------------------------------------------------------
c     INPUT:
c
c     wavek      : the Helmholtz coefficient
c     rscale     : scaling parameter used in forming expansion
c                                   (see h3dformmp1)
c     center     : coordinates of the expansion center
c     locexp     : coeffs of the j-expansion
c     nterms     : order of the h-expansion
c     nterms1    : order of the truncated expansion
c     ztarg(3)   : target vector
c     nt         : number of targets
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
c---------------------------------------------------------------------
c     OUTPUT:
c
c     ier        : error return code
c		      ier=0	returned successfully
c		      ier=8 insuffficient workspace 
c		      ier=16 insufficient memory 
c                            in subroutine "jfuns3d"
c     pot        : potential at ztarg (if requested)
c     fld(3)     : gradient at ztarg (if requested)
c
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer lwfjs
      real *8 center(3),ztarg(3,1)
      real *8, allocatable :: w(:)
      complex *16 wavek,pot(1),fld(3,1),pot0,fld0(3)
      complex *16 locexp(0:nterms,-nterms:nterms)
c
c ... Assigning work spaces for various temporary arrays:
c
      ier=0
c
      lwfjs=nterms+1000
      ipp=1
      lpp=(nterms+1)**2+3
      ippd  = ipp+lpp
c
      iephi=ippd+lpp
      lephi=2*(2*nterms+1)+7
c
      iiscale=iephi+lephi
      liscale=(lwfjs+1)+3
c
      ifjs=iiscale+liscale
      lfjs=2*(lwfjs+1)+3
c
      ifjder=ifjs+lfjs
      lfjder=2*(nterms+1)+3
c
      lused=ifjder+lfjder
      allocate(w(lused))
c
      do i=1,nt
      call h3dtaeval_trunc0(jer,wavek,rscale,center,locexp,
     1   nterms,nterms1,ztarg(1,i),
     1	     pot0,fld0,w(ipp),w(ippd),w(iephi),w(ifjs),
     1       w(ifjder),lwfjs,w(iiscale),wlege,nlege)
      pot(i)=pot(i)+pot0
      fld(1,i)=fld(1,i)+fld0(1)
      fld(2,i)=fld(2,i)+fld0(2)
      fld(3,i)=fld(3,i)+fld0(3)
      enddo
ccc      if (jer.ne.0) ier=16
c
      return
      end
c
c
c**********************************************************************
      subroutine h3dtaeval_trunc(wavek,rscale,center,locexp,nterms,
     $     nterms1,
     1		ztarg,pot,fld,wlege,nlege,ier)
c**********************************************************************
c
c     This subroutine evaluates a j-expansion centered at CENTER
c     at the target point ZTARG. 
c
c     pot =  sum sum  locexp(n,m) j_n(k r) Y_nm(theta,phi)
c             n   m
c
c---------------------------------------------------------------------
c     INPUT:
c
c     wavek      : the Helmholtz coefficient
c     rscale     : scaling parameter used in forming expansion
c                                   (see h3dformmp1)
c     center     : coordinates of the expansion center
c     locexp     : coeffs of the j-expansion
c     nterms     : order of the h-expansion
c     nterms1    : order of the truncated expansion
c     ztarg      : target vector
c     wlege  :    precomputed array of scaling coeffs for Pnm
c     nlege  :    dimension parameter for wlege
c---------------------------------------------------------------------
c     OUTPUT:
c
c     ier        : error return code
c		      ier=0	returned successfully
c		      ier=8 insuffficient workspace 
c		      ier=16 insufficient memory 
c                            in subroutine "jfuns3d"
c     pot        : potential at ztarg (if requested)
c     fld        : gradient at ztarg (if requested)
c
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer lwfjs
      real *8 center(3),ztarg(3)
      real *8, allocatable :: w(:)
      complex *16 wavek,pot,fld(3)
      complex *16 locexp(0:nterms,-nterms:nterms)
c
c ... Assigning work spaces for various temporary arrays:
c
      ier=0
c
      lwfjs=nterms+1000
      ipp=1
      lpp=(nterms+1)**2+3
      ippd  = ipp+lpp
c
      iephi=ippd+lpp
      lephi=2*(2*nterms+1)+7
c
      iiscale=iephi+lephi
      liscale=(lwfjs+1)+3
c
      ifjs=iiscale+liscale
      lfjs=2*(lwfjs+1)+3
c
      ifjder=ifjs+lfjs
      lfjder=2*(nterms+1)+3
c
      lused=ifjder+lfjder
      allocate(w(lused))
c
      call h3dtaeval_trunc0(jer,wavek,rscale,center,locexp,
     $   nterms,nterms1,ztarg,
     1	     pot,fld,w(ipp),w(ippd),w(iephi),w(ifjs),
     2       w(ifjder),lwfjs,w(iiscale),wlege,nlege)
      if (jer.ne.0) ier=16
c
      return
      end
c
c
c
c**********************************************************************
      subroutine h3dtaeval_trunc0(ier,wavek,rscale,center,locexp,
     $     nterms,nterms1,ztarg,
     $     pot,fld,pp,ppd,ephi,fjs,fjder,lwfjs,iscale,
     $     wlege,nlege)
c**********************************************************************
c
c     See h3dtaeval for comments.
c     (pp and ppd are storage arrays for Ynm and Ynm')
c
c----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer iscale(0:1)
      real *8 center(3),ztarg(3),zdiff(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 wavek,pot,fld(3),ephi1,ephi1inv
      complex *16 locexp(0:nterms,-nterms:nterms)
      complex *16 ephi(-nterms-1:nterms+1)
      complex *16 fjsuse,fjs(0:1),fjder(0:1)
c
      complex *16 eye,ur,utheta,uphi
      complex *16 ztmp,z
      complex *16 ztmp1,ztmp2,ztmp3,ztmpsum
      complex *16 ux,uy,uz
c
      data eye/(0.0d0,1.0d0)/
c
      ier=0
      done=1.0d0
c
      zdiff(1)=ztarg(1)-center(1)
      zdiff(2)=ztarg(2)-center(2)
      zdiff(3)=ztarg(3)-center(3)
c
c     Convert to spherical coordinates
c
      call cart2polar(zdiff,r,theta,phi)
      ctheta = dcos(theta)
      stheta=sqrt(done-ctheta*ctheta)
      cphi = dcos(phi)
      sphi = dsin(phi)
      ephi1 = dcmplx(cphi,sphi)
c
c     compute e^{eye*m*phi} array.
c
c
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=dconjg(ephi1)
      do i=2,nterms+1
         ephi(i)=ephi(i-1)*ephi1
         ephi(-i)=ephi(-i+1)*ephi(-1)
      enddo
c
c     compute coefficients in change of variables from spherical
c     to Cartesian gradients. In phix, phiy, we leave out the 
c     1/sin(theta) contribution, since we use values of Ynm (which
c     multiplies phix and phiy) that are scaled by 
c     1/sin(theta).
c
c     In thetax, thetaty, phix, phiy we leave out the 1/r factors in the 
c     change of variables to avoid blow-up at the origin.
c     For the n=0 mode, it is not relevant. For n>0 modes,
c     we use the recurrence relation 
c
c     (2n+1)fjs_n(kr)/(kr) = fjs(n+1)*rscale + fjs(n-1)/rscale
c
c     to avoid division by r. The variable fjsuse is set to fjs(n)/r:
c
c           fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
c	    fjsuse = wavek*fjsuse/(2*n+1.0d0)
c
c     
c
      rx = stheta*cphi
      thetax = ctheta*cphi
      phix = -sphi
      ry = stheta*sphi
      thetay = ctheta*sphi
      phiy = cphi
      rz = ctheta
      thetaz = -stheta
      phiz = 0.0d0
c
c     get the associated Legendre functions:
c
      call ylgndr2sfw(nterms1,ctheta,pp,ppd,wlege,nlege)
c
c     get the spherical Bessel functions and their derivatives.
c
      ifder=1
      z=wavek*r
      call jfuns3d(jer,nterms1,z,rscale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
      if (jer.ne.0) then
         ier=8
         return
      endif
c
c     scale derivatives of Bessel functions so that they are
c     derivatives with respect to r.
c
c
      pot=locexp(0,0)*fjs(0)
      do i=0,nterms1
         fjder(i)=fjder(i)*wavek
      enddo
      ur = locexp(0,0)*fjder(0)
      utheta = 0.0d0
      uphi = 0.0d0
c     
c     compute the potential and the field:
c
      do n=1,nterms1
         pot=pot+locexp(n,0)*fjs(n)*pp(n,0)
         ur = ur + fjder(n)*pp(n,0)*locexp(n,0)
         fjsuse = fjs(n+1)*rscale + fjs(n-1)/rscale
         fjsuse = wavek*fjsuse/(2*n+1.0d0)
         utheta = utheta -locexp(n,0)*fjsuse*ppd(n,0)*stheta
         do m=1,n
            ztmp1=fjs(n)*pp(n,m)*stheta
            ztmp2 = locexp(n,m)*ephi(m) 
            ztmp3 = locexp(n,-m)*ephi(-m)
            ztmpsum = ztmp2+ztmp3
            pot=pot+ztmp1*ztmpsum
            ur = ur + fjder(n)*pp(n,m)*stheta*ztmpsum
            utheta = utheta -ztmpsum*fjsuse*ppd(n,m)
            ztmpsum = eye*m*(ztmp2 - ztmp3)
            uphi = uphi + fjsuse*pp(n,m)*ztmpsum
         enddo
      enddo
      ux = ur*rx + utheta*thetax + uphi*phix
      uy = ur*ry + utheta*thetay + uphi*phiy
      uz = ur*rz + utheta*thetaz + uphi*phiz
      fld(1) = -ux
      fld(2) = -uy
      fld(3) = -uz
      return
      end
c
c
c
c
c
C***********************************************************************
      subroutine h3dformmp_trunc(ier,zk,scale,sources,charge,ns,center,
     1                  nterms,nterms1,mpole,wlege,nlege)
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
c		                   ier=0  returned successfully
c		                   ier=8  insufficient memory 
c		                   ier=16 insufficient memory 
c                                         in subroutine "jfuns3d"
c                                         called in h3dformmp1
c     mpole           : coeffs of the h-expansion
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,ns,i,l,m, ier, ier1, lused
      real *8 center(1),sources(3,ns)
      real *8 scale
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 eye,zk,charge(1)
      data eye/(0.0d0,1.0d0)/
C
C----- set mpole to zero
C
      do l = 0,nterms
         do m=-l,l
            mpole(l,m) = 0.0d0
         enddo
      enddo
c
      ier = 0
      do i = 1, ns
         call h3dformmp_trunc1
     $   (ier1,zk,scale,sources(1,i),charge(i),center,
     1        nterms,nterms1,mpole,wlege,nlege)
      enddo
      if (ier1.ne.0) ier = ier1
c
      do l = 0,nterms
         do m=-l,l
            mpole(l,m) = mpole(l,m)*eye*zk
         enddo
      enddo
C
      return
      end
C
C***********************************************************************
      subroutine h3dformmp_add_trunc
     $     (ier,zk,scale,sources,charge,ns,center,
     1                  nterms,nterms1,mpole,wlege,nlege)
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
c		                   ier=0  returned successfully
c		                   ier=8  insufficient memory 
c		                   ier=16 insufficient memory 
c                                         in subroutine "jfuns3d"
c                                         called in h3dformmp1
c    
c
c     mpole           : coeffs of the h-expansion
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer nterms,ns,i,l,m, ier, ier1, lused
      real *8 center(1),sources(3,ns)
      real *8 scale
      complex *16 mpole(0:nterms,-nterms:nterms)
      complex *16 eye,zk,charge(1)
      complex *16, allocatable :: mptemp(:,:)
      data eye/(0.0d0,1.0d0)/
C
C----- set mpole to zero
C
        allocate( mptemp(0:nterms,-nterms:nterms) )

        do l = 0,nterms
          do m=-l,l
             mptemp(l,m) = 0
          enddo
        enddo

        call h3dformmp_trunc
     $     (ier,zk,scale,sources,charge,ns,center,
     1     nterms,nterms1,mptemp,wlege,nlege)

      do l = 0,nterms
         do m=-l,l
            mpole(l,m) = mpole(l,m)+mptemp(l,m)
         enddo
      enddo
C
      return
      end
C
c**********************************************************************
      subroutine h3dformmp_trunc1(ier,zk,rscale,source,charge,center,
     1		nterms,nterms1,mpole,wlege,nlege)
c**********************************************************************
c
c     This subroutine creates the h-expansion about CENTER
c     due to a charge located at the point SOURCE.
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformmp0 below.
c
c-----------------------------------------------------------------------
c     INPUT:
c
c     zk      : the Helmholtz coefficient
c     rscale  : scaling parameter
c     source  : coordinates of the charge
c     charge  : complex charge strength
c     center  : coordinates of the expansion center
c     nterms  : order of the h-expansion
C     nterms1 : order of truncated expansion
c     wlege   :    precomputed array of scaling coeffs for Pnm
c     nlege   :    dimension parameter for wlege
c-----------------------------------------------------------------------
c     OUTPUT:
c
c     ier     : error return code
c		      ier=0 returned successfully
c		      ier=8	insufficient memory 
c		      ier=16 insufficient memory 
c                            in subroutine "jfuns3d"
c                            called in h3dformmp0
c                            
c     mpole   : coeffs of the h-expansion
c            
c     NOTE: Parameter lwfjs is set to nterms+1000
c           Should be sufficient for any Helmholtz parameter
c-----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer lwfjs
      real *8 source(3),center(3)
      real *8, allocatable :: w(:)
      complex *16 zk,mpole(0:nterms,-nterms:nterms)
      complex *16 charge
c
c ... Assign work spaces:
c
      ier=0
c
      lwfjs=nterms+1000
      ipp=1
      lpp=(nterms+1)**2+7
      ippd = ipp + lpp
c
      iephi=ippd+lpp
      lephi=2*(2*nterms+1)+7
c
      ifjder=iephi+lephi
      lfjder=2*(nterms+1)+7
c
      ifjs=ifjder+lfjder
      lfjs=2*(lwfjs+1)+7
c
      iiscale=ifjs+lfjs
      liscale=(lwfjs+1)+7
c
      lused=iiscale+liscale
      allocate(w(lused))
c
      call h3dformmp_trunc0(jer,zk,rscale,source,charge,center,
     $   nterms,nterms1,
     1		mpole,w(ipp),w(ippd),w(iephi),w(ifjs),lwfjs,
     2          w(iiscale),w(ifjder),wlege,nlege)
      if (jer.ne.0) ier=16
c
      return
      end
c
c
c
c**********************************************************************
      subroutine h3dformmp_trunc0(ier,zk,rscale,source,charge,center,
     1		nterms,nterms1,
     $     mpole,pp,ppd,ephi,fjs,lwfjs,iscale,fjder,wlege,nlege)
c**********************************************************************
c
c     See h3dformmp1 for comments.
c
c----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer iscale(0:1)
      real *8 source(3),center(3),zdiff(3)
      real *8 pp(0:nterms,0:nterms)
      real *8 ppd(0:nterms,0:nterms)
      complex *16 zk,mpole(0:nterms,-nterms:nterms)
      complex *16 charge
      complex *16 ephi(-nterms:nterms),ephi1,ephi1inv
      complex *16 fjs(0:1),ztmp,fjder(0:1),z
      data thresh/1.0d-15/
c
c
      ier=0
c
      zdiff(1)=source(1)-center(1)
      zdiff(2)=source(2)-center(2)
      zdiff(3)=source(3)-center(3)
c
      call cart2polar(zdiff,r,theta,phi)
      ctheta = dcos(theta)
      stheta=sqrt(1.0d0-ctheta*ctheta)
      cphi = dcos(phi)
      sphi = dsin(phi)
      ephi1 = dcmplx(cphi,sphi)
c
c     compute exp(eye*m*phi) array
c
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=dconjg(ephi1)
      do i=2,nterms+1
         ephi(i)=ephi(i-1)*ephi1
         ephi(-i)=ephi(-i+1)*ephi(-1)
      enddo
c
c     get the associated Legendre functions:
c
ccc      call ylgndr(nterms,ctheta,pp)
      call ylgndrfw(nterms1,ctheta,pp,wlege,nlege)
ccc      call ylgndr2s(nterms,ctheta,pp,ppd)
ccc      call prinf(' after ylgndr with nterms = *',nterms,1)
ccc      call prinm2(pp,nterms)
c
c     get Bessel functions:
c
      ifder=0
      z=zk*r
      call jfuns3d(jer,nterms1,z,rscale,fjs,ifder,fjder,
     1	      lwfjs,iscale,ntop)
c
c
c     multiply all jn by charge strength.
c
      do n = 0,nterms1
         fjs(n) = fjs(n)*charge
      enddo
      if (jer.ne.0) then
	 ier=16
	 return
      endif
c
c
c     Compute contribution to mpole coefficients.
c
c     Recall that there are multiple definitions of scaling for
c     Ylm. Using our standard definition, 
c     the addition theorem takes the simple form 
c
c        e^( i k r}/r = 
c         (ik) \sum_n \sum_m  j_n(k|S|) Ylm*(S) h_n(k|T|)Ylm(T)
c
c     so contribution is j_n(k|S|) times
c   
c       Ylm*(S)  = P_l,m * dconjg(ephi(m))               for m > 0   
c       Yl,m*(S)  = P_l,|m| * dconjg(ephi(m))            for m < 0
c                   
c       where P_l,m is the scaled associated Legendre function.
c
c     The factor (i*k) is taken care of after all source contributions
c     have been included in the calling subroutine h3dformmp.
c
      mpole(0,0)= mpole(0,0) + fjs(0)
      do n=1,nterms1
         dtmp=pp(n,0)
         mpole(n,0)= mpole(n,0) + dtmp*fjs(n)
         do m=1,n
cc            ztmp=stheta*pp(n,m)*fjs(n)
            ztmp=pp(n,m)*fjs(n)
            mpole(n, m)= mpole(n, m) + ztmp*dconjg(ephi(m))
            mpole(n,-m)= mpole(n,-m) + ztmp*dconjg(ephi(-m))
         enddo
      enddo
c

c
      return
      end
c
c
c
c
c
c
c**********************************************************************
      subroutine h3dformta_trunc
     $     (ier,zk,rscale,sources,charge,ns,center,
     1     nterms,nterms1,locexp,wlege,nlege)
c**********************************************************************
c
c     This subroutine creates a local (j) expansion about the point
c     CENTER due to the NS sources at the locations SOURCES(3,*).
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformta1/h3dformta0 below.
c
c----------------------------------------------------------------------
c     INPUT:
c
c     zk       : Helmholtz coefficient
c     rscale   : scaling parameter
c                     should be less than one in magnitude.
c                     Needed for low frequency regime only
c                     with rsclale abs(wavek) recommended.
c     sources   : coordinates of the sources
c     charge    : charge strengths
c     ns        : number of sources
c     center    : coordinates of the expansion center
c     nterms    : order of the j-expansion
C     nterms1 : order of truncated expansion
c     wlege   :    precomputed array of scaling coeffs for Pnm
c     nlege   :    dimension parameter for wlege
c----------------------------------------------------------------------
c     OUTPUT:
c
c     ier       : error return code
c		  ier=0	returned successfully;
c	 	  ier=4  d is out of range in h3dall
c
c     locexp    : coeffs for the j-expansion
c----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer ns
      real *8 sources(3,ns),center(3)
      complex *16 zk,locexp(0:nterms,-nterms:nterms), charge(ns)
      complex *16 eye
      data eye/(0.0d0,1.0d0)/
c
c     initialize local exp
c
      do l = 0,nterms
         do m = -l,l
            locexp(l,m) = 0.0d0
         enddo
      enddo
c
      do i = 1,ns
         call h3dformta_trunc1(ier,zk,rscale,sources(1,i),charge(i),
     1		center,nterms,nterms1,locexp,wlege,nlege)
      enddo
c
c     scale by (i*k)
c
      do l = 0,nterms
         do m=-l,l
            locexp(l,m) = locexp(l,m)*eye*zk
         enddo
      enddo
C
      return
      end
c
c
c**********************************************************************
      subroutine h3dformta_add_trunc
     $     (ier,zk,rscale,sources,charge,ns,center,
     1     nterms,nterms1,locexp,wlege,nlege)
c**********************************************************************
c
c     This subroutine creates a local (j) expansion about the point
c     CENTER due to the NS sources at the locations SOURCES(3,*).
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformta1/h3dformta0 below.
c
c----------------------------------------------------------------------
c     INPUT:
c
c     zk       : Helmholtz coefficient
c     rscale   : scaling parameter
c                     should be less than one in magnitude.
c                     Needed for low frequency regime only
c                     with rsclale abs(wavek) recommended.
c     sources   : coordinates of the sources
c     charge    : charge strengths
c     ns        : number of sources
c     center    : coordinates of the expansion center
c     nterms    : order of the j-expansion
c     nterms1   : order of the truncated expansion
c     wlege   :    precomputed array of scaling coeffs for Pnm
c     nlege   :    dimension parameter for wlege
c----------------------------------------------------------------------
c     OUTPUT:
c
c     ier       : error return code
c		  ier=0	returned successfully;
c		  ier=2	insufficient memory in workspace w
c	 	  ier=4  d is out of range in h3dall
c
c     locexp    : coeffs for the j-expansion
c----------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      integer ns
      real *8 sources(3,ns),center(3)
      complex *16 zk,locexp(0:nterms,-nterms:nterms), charge(ns)
      complex *16 eye
      complex *16, allocatable :: mptemp(:,:)
      data eye/(0.0d0,1.0d0)/
c
c     initialize local exp
c

        allocate( mptemp(0:nterms,-nterms:nterms) )
c
        do l = 0,nterms
          do m=-l,l
             mptemp(l,m) = 0
          enddo
        enddo
c
        call h3dformta_trunc
     $     (ier,zk,rscale,sources,charge,ns,center,
     1     nterms,nterms1,mptemp,wlege,nlege)
c
      do l = 0,nterms
         do m=-l,l
            locexp(l,m) = locexp(l,m)+mptemp(l,m)
         enddo
      enddo
C
      return
      end
c
c
c
c
c
c
c**********************************************************************
      subroutine h3dformta_trunc1(ier,wavek,rscale,source,charge,center,
     &		nterms,nterms1,locexp,wlege,nlege)
c**********************************************************************
c
c     This subroutine creates the local expansion about CENTER
c     due to a single charge located at SOURCE.
c     This is the memory management routine. Work is done in the
c     secondary call to h3dformta0 below.
c
c---------------------------------------------------------------------
c INPUT:
c
c     wavek     : the Helmholtz coefficient
c     rscale    : scaling parameter
c                         should be less than one in magnitude.
c                         Needed for low frequency regime only
c                         with rsclale abs(wavek) recommended.
c     source    : coordinates of the source
c     charge    : coordinates of the source
c     center    : coordinates of the expansion center
c     nterms    : order of the j-expansion
c     nterms1   : order of truncated expansion
c     wlege   :    precomputed array of scaling coeffs for Pnm
c     nlege   :    dimension parameter for wlege
c---------------------------------------------------------------------
c OUTPUT:
c
c     ier    : error return code
c	           ier=0 successful execution
c		   ier=2 insufficient memory in workspace w
c	 	   ier=4 d is out of range in h3dall
c     locexp : coefficients of the local expansion
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 source(3),center(3)
      real *8, allocatable :: w(:)
      complex *16 wavek,locexp(0:nterms,-nterms:nterms), charge
c
c     Carve up workspace
c
      ier=0
c
      ipp=1
      lpp=(nterms+1)**2+7
c
      iephi=ipp+lpp
      lephi=2*(2*nterms+1)+7
c
      ifhs=iephi+lephi
      lfhs=2*(nterms+1)+7
c
      lused=ifhs+lfhs
      allocate(w(lused))
c
      call h3dformta_trunc0(jer,wavek,rscale,source,charge,center,
     &		nterms,nterms1,locexp,w(ipp),w(iephi),w(ifhs),
     $   wlege,nlege)
      if (jer.ne.0) ier=4
c
      return
      end
c
c
c
c**********************************************************************
      subroutine h3dformta_trunc0(ier,wavek,rscale,source,charge,
     &		center,nterms,nterms1,locexp,pp,ephi,fhs,wlege,nlege)
c**********************************************************************
c
c     See h3dformta/h3dformta1 for comments
c
c---------------------------------------------------------------------
      implicit real *8 (a-h,o-z)
      real *8 source(3),center(3),zdiff(3)
      real *8 pp(0:nterms,0:nterms)
      complex *16 wavek,locexp(0:nterms,-nterms:nterms), charge
      complex *16 ephi(-nterms:nterms),ephi1,ephi1inv
      complex *16 fhs(0:nterms),ztmp,fhder(0:1),z
      data thresh/1.0d-15/
c
      ier=0
c
      zdiff(1)=source(1)-center(1)
      zdiff(2)=source(2)-center(2)
      zdiff(3)=source(3)-center(3)
c
      done=1
      call cart2polar(zdiff,r,theta,phi)
      ctheta = dcos(theta)
      stheta=sqrt(done-ctheta*ctheta)
      cphi = dcos(phi)
      sphi = dsin(phi)
      ephi1 = dcmplx(cphi,sphi)
c
c     Compute the e^{eye*m*phi} array
c
      ephi1inv=1.0d0/ephi1
c
      ephi(0)=1.0d0
      ephi(1)=ephi1
      ephi(-1)=ephi1inv
      do i=2,nterms
         ephi(i)=ephi(i-1)*ephi1
         ephi(-i)=ephi(-i+1)*ephi1inv
      enddo
c
c     get the Ynm
c
ccc      call ylgndr(nterms,ctheta,pp)
      call ylgndrfw(nterms,ctheta,pp,wlege,nlege)
c
c     compute Hankel functions and scale them by charge strength.
c
      ifder=0
      z=wavek*r
      if (abs(z).lt.thresh) then
         ier = 4
         return
      endif
      call h3dall(nterms1,z,rscale,fhs,ifder,fhder)
      do n = 0, nterms1
         fhs(n) = fhs(n)*charge
      enddo
c
c     Compute contributions to locexp
c
      locexp(0,0)=locexp(0,0) + fhs(0)
      do n=1,nterms1
         locexp(n,0)=locexp(n,0) + pp(n,0)*fhs(n)
         do m=1,n
            ztmp=pp(n,m)*fhs(n)
	    locexp(n,m)=locexp(n,m) + ztmp*ephi(-m)
	    locexp(n,-m)=locexp(n,-m) + ztmp*ephi(m)
         enddo
      enddo
      return
      end
