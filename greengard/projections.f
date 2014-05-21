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
      subroutine h3dmpevalspherenmstab_fast(Mrot,wavek,scalej,r,
     1           radius,ntrunc,ntermsj,ynm,ynmd,phitemp,phitempn,
     2           nquad,xnodes,fhs,fhder)
      implicit none
      integer l,m,n,ntermsj,ntrunc,nquad
      real *8 radius,r,ctheta,stheta,cthetaj,sthetaj,thetan,rj,rn
      real *8 scalei,scalej
      real *8 xnodes(nquad)
      real *8 ynm(0:ntrunc,0:ntrunc),ynmd(0:ntrunc,0:ntrunc)
      real *8 rat1(0:ntrunc,0:ntrunc),rat2(0:ntrunc,0:ntrunc)
      complex *16 Mrot(0:ntermsj,-ntermsj:ntermsj)
      complex *16 phitemp(nquad,-ntrunc:ntrunc)
      complex *16 phitempn(nquad,-ntrunc:ntrunc)
      complex *16 imag,wavek,z,ut1,ut2,ut3
      complex *16 fhs(0:ntrunc),fhder(0:ntrunc)
      data imag/(0.0d0,1.0d0)/
      do l=1,nquad
         do m=-ntrunc,ntrunc
            phitemp(l,m) = 0.0d0
            phitempn(l,m) = 0.0d0
         enddo
      enddo
      call ylgndrini(ntrunc,rat1,rat2)
      do l=1,nquad
	 ctheta=xnodes(l)
	 stheta=dsqrt(1.0d0-ctheta**2)
         rj=(r+radius*ctheta)**2+(radius*stheta)**2
         rj=dsqrt(rj)
	 cthetaj=(r+radius*ctheta)/rj
	 sthetaj=dsqrt(1.0d0-cthetaj**2)
	 rn=sthetaj*stheta+cthetaj*ctheta
	 thetan=(cthetaj*stheta-ctheta*sthetaj)/rj
	 z=wavek*rj
	 call ylgndr2sf(ntrunc,cthetaj,ynm,ynmd,rat1,rat2)
	 call h3dall(ntrunc,z,scalej,fhs,1,fhder)
         do n=0,ntrunc
	    fhder(n) = fhder(n)*wavek
         enddo
         do n=1,ntrunc
	    do m=1,n
  	       ynm(n,m)=ynm(n,m)*sthetaj
            enddo
         enddo
	 phitemp(l,0)=Mrot(0,0)*fhs(0)
	 phitempn(l,0)=Mrot(0,0)*fhder(0)*rn
	 do n=1,ntrunc
	    phitemp(l,0)=phitemp(l,0)+Mrot(n,0)*fhs(n)*ynm(n,0)
	    ut1=fhder(n)*rn
	    ut2=fhs(n)*thetan
	    ut3=ut1*ynm(n,0)-ut2*ynmd(n,0)*sthetaj
	    phitempn(l,0)=phitempn(l,0)+ut3*Mrot(n,0)
            do m=1,n
	       z=fhs(n)*ynm(n,m)
	       phitemp(l,m)=phitemp(l,m)+Mrot(n,m)*z
	       phitemp(l,-m)=phitemp(l,-m)+Mrot(n,-m)*z
	       ut3=ut1*ynm(n,m)-ut2*ynmd(n,m)
	       phitempn(l,m)=phitempn(l,m)+ut3*Mrot(n,m)
	       phitempn(l,-m)=phitempn(l,-m)+ut3*Mrot(n,-m)
	    enddo
	 enddo
      enddo
      return
      end
C***********************************************************************
      subroutine h3dprojlocsepstab_fast
     $          (nterms,ldl,nquadn,ntold,xnodes,wts,
     1           phitemp,phitempn,local,local2,ynm)
      implicit real *8 (a-h,o-z)
      integer nterms,nquadn,nquadm,ier
      integer l,m,jj,kk
      real *8 wts(1),xnodes(1)
      real *8 ynm(0:nterms,0:nterms)
      complex *16 wavek
      complex *16 local(0:ldl,-ldl:ldl)
      complex *16 local2(0:ldl,-ldl:ldl)
      complex *16 phitemp(nquadn,-ntold:ntold)
      complex *16 phitempn(nquadn,-ntold:ntold)
      complex *16 ephi,imag,emul,sum,zmul,emul1
      real *8 rat1(0:nterms,0:nterms),rat2(0:nterms,0:nterms)
      data imag/(0.0d0,1.0d0)/
      do l = 0,ldl
         do m = -l,l
	    local(l,m) = 0.0d0
	    local2(l,m) = 0.0d0
         enddo
      enddo
      call ylgndrini(nterms,rat1,rat2)
      do jj=1,nquadn
	 cthetaj = xnodes(jj)
	 call ylgndrf(nterms,cthetaj,ynm,rat1,rat2)
         do m=-ntold,ntold
	    zmul = phitemp(jj,m)*wts(jj)/2.0d0
            do n=abs(m),nterms
               local(n,m) = local(n,m) + 
     1   	       zmul*ynm(n,abs(m))
            enddo
	    zmul = phitempn(jj,m)*wts(jj)/2.0d0
            do n=abs(m),nterms
               local2(n,m) = local2(n,m) + 
     1   	       zmul*ynm(n,abs(m))
            enddo
         enddo
      enddo
      return
      end
