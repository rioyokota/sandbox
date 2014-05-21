C***********************************************************************
      subroutine h3drescalestab(nterms,lmp,local,localn,
     1           radius,wavek0,scale,fjs,fjder,nbessel,ier)
      implicit real *8 (a-h,o-z)
      integer nterms,ier
      integer l,m,jj,kk
      integer lwfhs
      complex *16 fjs(0:nbessel)
      complex *16 fjder(0:nterms)
      complex *16 local(0:lmp,-lmp:lmp)
      complex *16 localn(0:lmp,-lmp:lmp)
      complex *16 ephi,imag,emul,sum,zmul
      complex *16 wavek0,z,zh,zhn
      data imag/(0.0d0,1.0d0)/
      z = wavek0*radius
      call jfuns3d(ier1,nterms,z,scale,fjs,1,fjder,nbessel)
      if (ier1.eq.8) then
         ier = 8
	 return
      endif
      do l=0,nterms
         do m=-l,l
	    zh = fjs(l)
	    zhn = fjder(l)*wavek0
	    zmul = zh*zh + zhn*zhn
	    local(l,m) = (zh*local(l,m) + zhn*localn(l,m))/zmul
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
      subroutine h3dlocevalspherestab_fast(local,wavek,scale,zshift,
     1     radius,nterms,nterms2,lmp,ynm,ynmd,phitemp,phitempn,
     1     nquad,xnodes,fjs,fjder,nbessel,ier)
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
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      ier = 0
      do jj=1,nquad
      do m=-nterms2,nterms2
         phitemp(jj,m) = 0.0d0
         phitempn(jj,m) = 0.0d0
      enddo
      enddo
      call ylgndrini(nterms,rat1,rat2)
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
	 call ylgndr2sf(nterms,cthetaj,ynm,ynmd,rat1,rat2)
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
