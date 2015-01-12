      subroutine buildTree(Xj,numBodies,ncrit,
     1     nboxes,permutation,nlev,center,size)
      use arrays, only : listOffset,lists,levelOffset,nodes,boxes,
     1     centers,corners
      implicit none
      integer i,j,numBodies,ncrit,nboxes,nlev
      integer permutation(*)
      real *8 size
      real *8 Xj(3,*),center(3)
      do i=1,numBodies
         permutation(i)=i
      enddo
      allocate(nodes(20,numBodies))
      call growTree(Xj,numBodies,ncrit,nodes,
     1     nboxes,permutation,nlev,center,size)
      allocate(listOffset(nboxes,5))
      allocate(lists(2,189*nboxes))
      allocate(boxes(20,nboxes))
      allocate(centers(3,nboxes))
      allocate(corners(3,8,nboxes))
      do i=1,nboxes
         do j=1,20
            boxes(j,i)=nodes(j,i)
         enddo
      enddo
      call setCenter(center,size,nboxes)
      call getLists(nboxes)
      return
      end

      subroutine getNumChild(box,nkids)
      implicit none
      integer nkids,ikid
      integer box(20)
      nkids=0
      do ikid=1,box(7)
         if( box(6)+ikid-1 .ne. 0 ) nkids=nkids+1
      enddo
      return
      end

      subroutine getCell(ibox,box,nboxes,center,corners)
      use arrays, only : boxes,listOffset
      implicit none
      integer ibox,nboxes,i
      integer box(20)
      real *8 center(3),corners(3,8)
      if( (ibox.lt.1).or.(ibox.gt.nboxes) ) then
         print*,"Error: ibox out of bounds"
         stop
      endif
      do i=1,20
         box(i)=boxes(i,ibox)
      enddo
      call getCenter(ibox,center,corners)
      return
      end

      subroutine getLists(nboxes)
      use arrays, only : boxes,listOffset,corners
      implicit none
      integer i,j,k,ibox,jbox,nboxes,iparent,nkids,icoll,ncolls,kid
      integer nlist,ifinter
      integer kids(50000),parents(2000),list5(20000),stack(60000)
      do k=1,5
         do i=1,nboxes
            listOffset(i,k)=-1
         enddo
      enddo
c     construct lists 5,2 for all boxes
      do ibox=2,nboxes
         iparent=boxes(5,ibox)
         parents(1)=iparent
         call getList(5,iparent,parents(2),ncolls)
         ncolls=ncolls+1
         nkids=0
         do i=1,ncolls
            icoll=parents(i)
            do j=1,boxes(7,icoll)
               kid=boxes(6,icoll)+j-1
               if(kid.gt.0)then
                  if(kid.ne.ibox)then
                     nkids=nkids+1
                     kids(nkids)=kid
                  endif
               endif
            enddo
         enddo
c     sort the kids of the parent's collegues into the
c     lists 2, 5 of the box ibox
         do i=1,nkids
            kid=kids(i)
            call intersect(corners(1,1,kid),corners(1,1,ibox),ifinter)
            if(ifinter.eq.1)
     1           call setList(5,ibox,kid)
            if(ifinter.eq.0)
     1           call setList(2,ibox,kid)
         enddo
      enddo
c     now, construct lists 1, 3
      do i=1,nboxes
         if(boxes(6,i).le.0.and.boxes(1,i).ne.0)then
            call getList(5,i,list5,nlist)
            do j=1,nlist
               jbox=list5(j)
               call getList13(i,jbox,stack)
            enddo
         endif
      enddo
      do ibox=1,nboxes
         call getList(3,ibox,list5,nlist)
         do j=1,nlist
            call setList(4,list5(j),ibox)
         enddo
      enddo
      return
      end

      subroutine getList13(ibox,jbox0,stack)
      use arrays, only : boxes,corners
      implicit none
      integer jbox,jbox0,istack,nchilds,j,ijk,ibox,ifinter
      integer stack(3,*)
      jbox=jbox0
      istack=1
      stack(1,1)=1
      stack(2,1)=jbox
      nchilds=0
      do j=1,8
         if(boxes(6,jbox)-j+1.gt.0) nchilds=nchilds+1
      enddo
      stack(3,1)=nchilds
c     . . . move up and down the stack, generating the elements
c     of lists 1, 3 for the box jbox, as appropriate
      do ijk=1,1 000 000 000
c     if this box is separated from ibox - store it in list 3;
c     enter this fact in the parent's table; pass control
c     to the parent
         call intersect(corners(1,1,ibox),corners(1,1,jbox),ifinter)
         if(ifinter .eq. 1) exit
         call setList(3,ibox,jbox)
c     if storage capacity has been exceeed - bomb
         istack=istack-1
         stack(3,istack)=stack(3,istack)-1
         jbox=stack(2,istack)
      enddo
c     this box is not separated from ibox. if it is childless
c     - enter it in list 1; enter this fact in the parent's table;
c     pass control to the parent
      if(boxes(6,jbox) .eq. 0) then
         call setList(1,ibox,jbox)
c     . . . entered jbox in the list1 of ibox; if jbox
c     is on the finer level than ibox - enter ibox
c     in the list 1 of jbox
         if(boxes(1,jbox) .ne. boxes(1,ibox)) then
            call setList(1,jbox,ibox)
         endif
c     if we have processed the whole box jbox0, get out
c     of the subroutine
         if(jbox .eq. jbox0) return
         istack=istack-1
         stack(3,istack)=stack(3,istack)-1
         jbox=stack(2,istack)
      endif
      return
      end

      subroutine intersect(c1,c2,ifinter)
      implicit none
      integer i,ifinter
      real *8 xmin1, ymin1, zmin1, xmax1, ymax1, zmax1
      real *8 xmin2, ymin2, zmin2, xmax2, ymax2, zmax2
      real *8 eps
      real *8 c1(3,8),c2(3,8)
      xmin1=c1(1,1)
      ymin1=c1(2,1)
      zmin1=c1(3,1)
      xmax1=c1(1,1)
      ymax1=c1(2,1)
      zmax1=c1(3,1)
      xmin2=c2(1,1)
      ymin2=c2(2,1)
      zmin2=c2(3,1)
      xmax2=c2(1,1)
      ymax2=c2(2,1)
      zmax2=c2(3,1)
      do i=1,8
         if(xmin1 .gt. c1(1,i)) xmin1=c1(1,i)
         if(ymin1 .gt. c1(2,i)) ymin1=c1(2,i)
         if(zmin1 .gt. c1(3,i)) zmin1=c1(3,i)
         if(xmax1 .lt. c1(1,i)) xmax1=c1(1,i)
         if(ymax1 .lt. c1(2,i)) ymax1=c1(2,i)
         if(zmax1 .lt. c1(3,i)) zmax1=c1(3,i)
         if(xmin2 .gt. c2(1,i)) xmin2=c2(1,i)
         if(ymin2 .gt. c2(2,i)) ymin2=c2(2,i)
         if(zmin2 .gt. c2(3,i)) zmin2=c2(3,i)
         if(xmax2 .lt. c2(1,i)) xmax2=c2(1,i)
         if(ymax2 .lt. c2(2,i)) ymax2=c2(2,i)
         if(zmax2 .lt. c2(3,i)) zmax2=c2(3,i)
      enddo
      eps=xmax1-xmin1
      if(eps .gt. ymax1-ymin1) eps=ymax1-ymin1
      if(eps .gt. zmax1-zmin1) eps=zmax1-zmin1
      if(eps .gt. xmax2-xmin2) eps=xmax2-xmin2
      if(eps .gt. ymax2-ymin2) eps=ymax2-ymin2
      if(eps .gt. zmax2-zmin2) eps=zmax2-zmin2
      eps=eps/10000
      ifinter=1
      if(xmin1 .gt. xmax2+eps) ifinter=0
      if(xmin2 .gt. xmax1+eps) ifinter=0
      if(ymin1 .gt. ymax2+eps) ifinter=0
      if(ymin2 .gt. ymax1+eps) ifinter=0
      if(zmin1 .gt. zmax2+eps) ifinter=0
      if(zmin2 .gt. zmax1+eps) ifinter=0
      return
      end

      subroutine getCenter(ibox,center,corner)
      use arrays, only : centers,corners
      implicit none
      integer i,ibox
      real *8 center(3),corner(3,8)
      center(1)=centers(1,ibox)
      center(2)=centers(2,ibox)
      center(3)=centers(3,ibox)
      do i=1,8
         corner(1,i)=corners(1,i,ibox)
         corner(2,i)=corners(2,i,ibox)
         corner(3,i)=corners(3,i,ibox)
      enddo
      return
      end

      subroutine growTree(Xj,numBodies,ncrit,boxes,
     1     nboxes,permutation,nlev,center0,size)
      use arrays, only : levelOffset
      implicit none
      integer i,nlev,level,nlevChild
      integer iparent,nchild,ibody,nbody,ncrit,ii,jj,kk,numBodies
      integer offset,nboxes
      integer boxes(20,*),permutation(*),iwork(numBodies),nbody8(8)
      real *8 xmin,xmax,ymin,ymax,zmin,zmax
      real *8 size,sizey,sizez
      real *8 Xj(3,*),center0(3),center(3)
      xmin=Xj(1,1)
      xmax=Xj(1,1)
      ymin=Xj(2,1)
      ymax=Xj(2,1)
      zmin=Xj(3,1)
      zmax=Xj(3,1)
      do i=1,numBodies
         if(Xj(1,i) .lt. xmin) xmin=Xj(1,i)
         if(Xj(1,i) .gt. xmax) xmax=Xj(1,i)
         if(Xj(2,i) .lt. ymin) ymin=Xj(2,i)
         if(Xj(2,i) .gt. ymax) ymax=Xj(2,i)
         if(Xj(3,i) .lt. zmin) zmin=Xj(3,i)
         if(Xj(3,i) .gt. zmax) zmax=Xj(3,i)
      enddo
      size=xmax-xmin
      sizey=ymax-ymin
      sizez=zmax-zmin
      if(sizey .gt. size) size=sizey
      if(sizez .gt. size) size=sizez
      center0(1)=(xmin+xmax)/2
      center0(2)=(ymin+ymax)/2
      center0(3)=(zmin+zmax)/2
      boxes(1,1)=0 ! level
      boxes(2,1)=0 ! iX(1)
      boxes(3,1)=0 ! iX(2)
      boxes(4,1)=0 ! iX(3)
      boxes(5,1)=0 ! iparent
      boxes(6,1)=0 ! ichild
      boxes(7,1)=0 ! nchild
      boxes(8,1)=1 ! ibody
      boxes(9,1)=numBodies ! nbody
      levelOffset(1)=1
      levelOffset(2)=2
      do i=1,numBodies
         permutation(i)=i
      enddo
      nboxes=1
      nlev=0
      do level=1,100
         nlevChild=0
         do iparent=levelOffset(level),levelOffset(level+1)-1
            nbody=boxes(9,iparent)
            if(nbody.le.ncrit) cycle
            ii=boxes(2,iparent)
            jj=boxes(3,iparent)
            kk=boxes(4,iparent)
            call findCenter(center0,size,level,ii,jj,kk,center)
            ibody=boxes(8,iparent)
            call reorder(center,Xj,permutation(ibody),nbody,iwork,
     1           nbody8)
            nchild=0
            offset=ibody
            do i=0,7
               if(nbody8(i+1).eq.0) cycle
               nlevChild=nlevChild+1
               nboxes=nboxes+1
               nlev=level
               if(nboxes.gt.numBodies) then
                  print*,"Error: ibox out of bounds"
                  stop
               endif
               boxes(1,nboxes)=level
               boxes(2,nboxes)=ii*2+mod(i,2)
               boxes(3,nboxes)=jj*2+mod(i/2,2)
               boxes(4,nboxes)=kk*2+i/4
               boxes(5,nboxes)=iparent
               boxes(6,nboxes)=0
               boxes(7,nboxes)=0
               boxes(8,nboxes)=offset
               boxes(9,nboxes)=nbody8(i+1)
               nchild=nchild+1
               offset = offset + nbody8(i+1)
            enddo
            boxes(6,iparent)=nboxes-nchild+1
            boxes(7,iparent)=nchild
         enddo
         levelOffset(level+2)=levelOffset(level+1)+nlevChild
         if(nlevChild.eq.0) exit
      enddo
      return
      end

      subroutine setCenter(center0,size,nboxes)
      use arrays, only : boxes,centers,corners
      implicit none
      integer i,nboxes,level,ii,jj,kk
      real *8 x00,y00,z00,side,side2,size
      real *8 center(3),center0(3)
c     this subroutine produces arrays of centers and
c     corners for all boxes in the oct-tree structure.
c     input parameters:
c     center0 - the center of the box on the level 0, containing
c     the whole simulation
c     size - the side of the box on the level 0
c     boxes - an integer array dimensioned (10,nboxes)
c     column describes one box, as follows:
c     nboxes - the total number of boxes created
c     output parameters:
c     centers - the centers of all boxes in the array boxes
c     corners - the corners of all boxes in the array boxes
      x00=center0(1)-size/2
      y00=center0(2)-size/2
      z00=center0(3)-size/2
      do i=1,nboxes
         level=boxes(1,i)
         side=size/2**level
         side2=side/2
         ii=boxes(2,i)
         jj=boxes(3,i)
         kk=boxes(4,i)
         center(1)=x00+ii*side+side2
         center(2)=y00+jj*side+side2
         center(3)=z00+kk*side+side2
         centers(1,i)=center(1)
         centers(2,i)=center(2)
         centers(3,i)=center(3)

         corners(1,1,i)=center(1)-side/2
         corners(1,2,i)=corners(1,1,i)
         corners(1,3,i)=corners(1,1,i)
         corners(1,4,i)=corners(1,1,i)
         corners(1,5,i)=corners(1,1,i)+side
         corners(1,6,i)=corners(1,5,i)
         corners(1,7,i)=corners(1,5,i)
         corners(1,8,i)=corners(1,5,i)

         corners(2,1,i)=center(2)-side/2
         corners(2,2,i)=corners(2,1,i)
         corners(2,5,i)=corners(2,1,i)
         corners(2,6,i)=corners(2,1,i)
         corners(2,3,i)=corners(2,1,i)+side
         corners(2,4,i)=corners(2,3,i)
         corners(2,7,i)=corners(2,3,i)
         corners(2,8,i)=corners(2,3,i)

         corners(3,1,i)=center(3)-side/2
         corners(3,3,i)=corners(3,1,i)
         corners(3,5,i)=corners(3,1,i)
         corners(3,7,i)=corners(3,1,i)
         corners(3,2,i)=corners(3,1,i)+side
         corners(3,4,i)=corners(3,2,i)
         corners(3,6,i)=corners(3,2,i)
         corners(3,8,i)=corners(3,2,i)
      enddo
      return
      end

      subroutine findCenter(center0,size,level,i,j,k,center)
      implicit none
      integer level0,level,i,j,k
      real *8 side,side2,size,x0,y0,z0
      real *8 center(3),center0(3)
c     this subroutine finds the center of the box
c     number (i,j) on the level level. note that the
c     box on level 0 is assumed to have the center
c     center0, and the side size
      side=size/2**(level-1)
      side2=side/2
      x0=center0(1)-size/2
      y0=center0(2)-size/2
      z0=center0(3)-size/2
      level0=level
      center(1)=x0+i*side+side2
      center(2)=y0+j*side+side2
      center(3)=z0+k*side+side2
      return
      end

      subroutine reorder(center,z,iz,n,iwork,nbody)
      implicit none
      integer n1,n2,n3,n4,n5,n6,n7,n8,n12,n34,n56,n78
      integer n1234,n5678,n
      integer iz(*),iwork(*),nbody(*)
      real *8 center(3),z(3,*)
      n1=0
      n2=0
      n3=0
      n4=0
      n5=0
      n6=0
      n7=0
      n8=0
      n12=0
      n34=0
      n56=0
      n78=0
      n1234=0
      n5678=0
      call divide(z,iz,n,3,center(3),iwork,n1234)
      n5678=n-n1234
      if(n1234 .ne. 0)
     1     call divide(z,iz,n1234,2,center(2),iwork,n12)
      n34=n1234-n12
      if(n5678 .ne. 0)
     1     call divide(z,iz(n1234+1),n5678,2,center(2),iwork,n56)
      n78=n5678-n56
      if(n12 .ne. 0)
     1     call divide(z,iz,n12,1,center(1),iwork,n1)
      n2=n12-n1
      if(n34 .ne. 0)
     1     call divide(z,iz(n12+1),n34,1,center(1),iwork,n3)
      n4=n34-n3
      if(n56 .ne. 0)
     1     call divide(z,iz(n1234+1),n56,1,center(1),iwork,n5)
      n6=n56-n5
      if(n78 .ne. 0)
     1     call divide(z,iz(n1234+n56+1),n78,1,center(1),iwork,n7)
      n8=n78-n7
      nbody(1)=n1
      nbody(2)=n2
      nbody(3)=n3
      nbody(4)=n4
      nbody(5)=n5
      nbody(6)=n6
      nbody(7)=n7
      nbody(8)=n8
      return
      end

      subroutine divide(z,iz,n,d,thresh,iwork,n1)
      implicit none
      integer i1,i2,i,j,n1,n,d
      integer iz(*),iwork(*)
      real *8 thresh
      real *8 z(3,*)
      i1=0
      i2=0
      do i=1,n
         j=iz(i)
         if(z(d,j).le.thresh) then
            i1=i1+1
            iz(i1)=j
            cycle
         endif
         i2=i2+1
         iwork(i2)=j
      enddo
      do i=1,i2
         iz(i1+i)=iwork(i)
      enddo
      n1=i1
      return
      end

      subroutine setList(itype,ibox,list)
      use arrays, only : listOffset,lists
      implicit none
      integer ilast,ibox,itype,list,numele/0/
      ilast=listOffset(ibox,itype)
      numele=numele+1
      lists(1,numele)=ilast
      lists(2,numele)=list
      ilast=numele
      listOffset(ibox,itype)=ilast
      return
      end

      subroutine getList(itype,ibox,list,nlist)
      use arrays, only : listOffset,lists
      implicit none
      integer ilast,ibox,itype,nlist,i,j
      integer list(*)
      ilast=listOffset(ibox,itype)
      nlist=0
      do while(ilast.gt.0)
         if(lists(2,ilast).gt.0)then
            nlist=nlist+1
            list(nlist)=lists(2,ilast)
         endif
         ilast=lists(1,ilast)
      enddo
      do i=1,nlist/2
         j=list(i)
         list(i)=list(nlist-i+1)
         list(nlist-i+1)=j
      enddo
      return
      end
