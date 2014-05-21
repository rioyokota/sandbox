C***********************************************************************
      subroutine h3drescalestab(ntrunc,Lnm,Lnmd,
     1           radius,wavek,scalei,jn,jnd,nbessel,ier)
      implicit none
      integer n,m,ntrunc,nbessel,ier
      real *8 radius,scalei
      complex *16 jn(0:nbessel)
      complex *16 jnd(0:ntrunc)
      complex *16 Lnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Lnmd(0:ntrunc,-ntrunc:ntrunc)
      complex *16 imag,wavek,z,zh,zhn
      data imag/(0.0d0,1.0d0)/
      z = wavek*radius
      call jfuns3d(ier,ntrunc,z,scalei,jn,1,jnd,nbessel)
      do n=0,ntrunc
         do m=-n,n
	    zh=jn(n)
	    zhn=jnd(n)*wavek
	    z=zh*zh+zhn*zhn
	    Lnm(n,m)=(zh*Lnm(n,m)+zhn*Lnmd(n,m))/z
         enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dlocevalspherestab(local,wavek,scale,zshift,radius,
     1           nterms,nterms2,lmp,ynm,ynmd,phitemp,phitempn,
     2           nquad,xnodes,fjs,fjder,nbessel,ier)
      implicit real *8 (a-h,o-z)
      integer nterms
      integer l,m
      real *8 zshift
      real *8 xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      real *8 ynmd(0:nterms,0:nterms)
      complex *16 local(0:lmp,-lmp:lmp)
      complex *16 phitemp(nquad,-nterms2:nterms2)
      complex *16 phitempn(nquad,-nterms2:nterms2)
      complex *16 imag,pot,fld(3), wavek,z,uval,unval,ur,utheta
      complex *16 ephi1,ephik,ephi,fjs(0:nbessel),fjder(0:nbessel),ztmp1
      complex *16 ut1,ut2,ut3
      data imag/(0.0d0,1.0d0)/
      ier = 0
      do jj=1,nquad
      do m=-nterms2,nterms2
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (zshift+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (zshift+radius*ctheta)/rj
	 sthetaj = dsqrt(1.0d0-cthetaj**2)
	 rn = sthetaj*stheta + cthetaj*ctheta
	 thetan = (cthetaj*stheta - sthetaj*ctheta)/rj
	 z = wavek*rj
	 call ylgndr2s(nterms,cthetaj,ynm,ynmd)
	 call jfuns3d(jer,nterms,z,scale,fjs,1,fjder,nbessel)
         if (jer.eq.8) then
            ier = 8
	    return
         endif
	 do n = 0,nterms
	    fjder(n) = fjder(n)*wavek
         enddo
	 do n = 1,nterms
	    do m = 1,n
	       ynm(n,m) = ynm(n,m)*sthetaj
            enddo
         enddo
	 phitemp(jj,0) = local(0,0)*fjs(0)
	 phitempn(jj,0) = local(0,0)*fjder(0)*rn
         do n=1,nterms
	    phitemp(jj,0) = phitemp(jj,0) +
     1                local(n,0)*fjs(n)*ynm(n,0)
	    ut1 = fjder(n)*rn
	    ut2 = fjs(n)*thetan
	    ut3 = ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(jj,0) = phitempn(jj,0)+ut3*local(n,0)
	    do m=1,min(n,nterms2)
	       ztmp1 = fjs(n)*ynm(n,m)
	       phitemp(jj,m) = phitemp(jj,m) +
     1                local(n,m)*ztmp1
	       phitemp(jj,-m) = phitemp(jj,-m) +
     1                local(n,-m)*ztmp1
	       ut3 = ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(jj,m) = phitempn(jj,m)+ut3*local(n,m)
	       phitempn(jj,-m) = phitempn(jj,-m)+ut3*local(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dlocevalspherestab_fast(Lrot,wavek,scalej,r,
     1     radius,ntermsj,ntermsi,ynm,ynmd,phitemp,phitempn,
     1     nquad,xnodes,jn,jnd,nbessel,ier)
      implicit real *8 (a-h,o-z)
      integer ntermsj
      integer l,m
      real *8 r
      real *8 xnodes(1)
      real *8 ynm(0:ntermsj,0:ntermsj)
      real *8 ynmd(0:ntermsj,0:ntermsj)
      complex *16 Lrot(0:ntermsi,-ntermsi:ntermsi)
      complex *16 phitemp(nquad,-ntermsi:ntermsi)
      complex *16 phitempn(nquad,-ntermsi:ntermsi)
      complex *16 imag,pot,fld(3), wavek,z,uval,unval,ur,utheta
      complex *16 ephi1,ephik,ephi,jn(0:nbessel),jnd(0:nbessel),ztmp1
      complex *16 ut1,ut2,ut3
      real *8 rat1(0:ntermsj,0:ntermsj),rat2(0:ntermsj,0:ntermsj)
      data imag/(0.0d0,1.0d0)/
      ier = 0
      do jj=1,nquad
      do m=-ntermsi,ntermsi
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      call ylgndrini(ntermsj,rat1,rat2)
      do jj=1,nquad
	 ctheta = xnodes(jj)
	 stheta = dsqrt(1.0d0 - ctheta**2)
         rj = (r+ radius*ctheta)**2 + (radius*stheta)**2
         rj = dsqrt(rj)
	 cthetaj = (r+radius*ctheta)/rj
	 sthetaj = dsqrt(1.0d0-cthetaj**2)
	 rn = sthetaj*stheta + cthetaj*ctheta
	 thetan = (cthetaj*stheta - sthetaj*ctheta)/rj
	 z = wavek*rj
	 call ylgndr2sf(ntermsj,cthetaj,ynm,ynmd,rat1,rat2)
	 call jfuns3d(jer,ntermsj,z,scalej,jn,1,jnd,nbessel)
         if (jer.eq.8) then
            ier = 8
	    return
         endif
	 do n = 0,ntermsj
	    jnd(n) = jnd(n)*wavek
         enddo
	 do n = 1,ntermsj
	    do m = 1,n
	       ynm(n,m) = ynm(n,m)*sthetaj
            enddo
         enddo
	 phitemp(jj,0) = Lrot(0,0)*jn(0)
	 phitempn(jj,0) = Lrot(0,0)*jnd(0)*rn
         do n=1,ntermsj
	    phitemp(jj,0) = phitemp(jj,0) +
     1                Lrot(n,0)*jn(n)*ynm(n,0)
	    ut1 = jnd(n)*rn
	    ut2 = jn(n)*thetan
	    ut3 = ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(jj,0) = phitempn(jj,0)+ut3*Lrot(n,0)
	    do m=1,min(n,ntermsi)
	       ztmp1 = jn(n)*ynm(n,m)
	       phitemp(jj,m) = phitemp(jj,m) +
     1                Lrot(n,m)*ztmp1
	       phitemp(jj,-m) = phitemp(jj,-m) +
     1                Lrot(n,-m)*ztmp1
	       ut3 = ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(jj,m) = phitempn(jj,m)+ut3*Lrot(n,m)
	       phitempn(jj,-m) = phitempn(jj,-m)+ut3*Lrot(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dprojlocsepstab_fast(ntrunc,nquad,xnodes,wts,
     1     phitemp,phitempn,Lnm,Lnmd,ynm)
      implicit none
      integer l,m,n,mabs,ntrunc,nquad
      real *8 cthetaj
      real *8 xnodes(nquad),wts(nquad)
      real *8 ynm(0:ntrunc,0:ntrunc)
      complex *16 Lnm(0:ntrunc,-ntrunc:ntrunc)
      complex *16 Lnmd(0:ntrunc,-ntrunc:ntrunc)
      complex *16 phitemp(nquad,-ntrunc:ntrunc)
      complex *16 phitempn(nquad,-ntrunc:ntrunc)
      complex *16 imag,wavek,z
      real *8 rat1(0:ntrunc,0:ntrunc),rat2(0:ntrunc,0:ntrunc)
      data imag/(0.0d0,1.0d0)/
      do n=0,ntrunc
         do m=-n,n
	    Lnm(n,m)=0.0d0
	    Lnmd(n,m)=0.0d0
         enddo
      enddo
      call ylgndrini(ntrunc,rat1,rat2)
      do l=1,nquad
	 cthetaj=xnodes(l)
	 call ylgndrf(ntrunc,cthetaj,ynm,rat1,rat2)
         do m=-ntrunc,ntrunc
            mabs=abs(m)
	    z=phitemp(l,m)*wts(l)/2.0d0
            do n=mabs,ntrunc
               Lnm(n,m)=Lnm(n,m)+z*ynm(n,mabs)
            enddo
	    z=phitempn(l,m)*wts(l)/2.0d0
            do n=mabs,ntrunc
               Lnmd(n,m)=Lnmd(n,m)+z*ynm(n,mabs)
            enddo
         enddo
      enddo
      return
      end
