      subroutine buildTree(Xj,numBodies,ncrit,
     $     nboxes,isource,nlev,center,size)
      use arrays, only : listOffset,lists,levelOffset,nodes,boxes,
     $     centers,corners
      implicit none
      integer i,j,numBodies,ncrit,nboxes,nlev
      integer isource(*)
      real *8 size
      real *8 Xj(3,*),center(3)
      do i=1,numBodies
         isource(i)=i
      enddo
      allocate(nodes(20,numBodies))
      call growTree(Xj,numBodies,ncrit,nodes,numBodies,
     $     nboxes,isource,nlev,center,size,0,100)
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
      do ikid=1,8
         if( box(5+ikid) .ne. 0 ) nkids=nkids+1
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
            do j=1,8
               kid=boxes(5+j,icoll)
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
     $           call setList(5,ibox,nboxes,kid,1)
            if(ifinter.eq.0)
     $           call setList(2,ibox,nboxes,kid,1)
         enddo
      enddo
c     now, construct lists 1, 3
      do i=1,nboxes
         if(boxes(6,i).le.0.and.boxes(1,i).ne.0)then
            call getList(5,i,list5,nlist)
            do j=1,nlist
               jbox=list5(j)
               call getList13(i,jbox,nboxes,stack)
            enddo
         endif
      enddo
      do ibox=1,nboxes
         call getList(3,ibox,list5,nlist)
         do j=1,nlist
            call setList(4,list5(j),nboxes,ibox,1)
         enddo
      enddo
      return
      end

      subroutine getList13(ibox,jbox0,nboxes,stack)
      use arrays, only : boxes,corners
      implicit real *8 (a-h,o-z)
      integer stack(3,*)
      jbox=jbox0
      istack=1
      stack(1,1)=1
      stack(2,1)=jbox
      nchilds=0
      do j=6,13
         if(boxes(j,jbox) .gt. 0) nchilds=nchilds+1
      enddo
      stack(3,1)=nchilds
c     . . . move up and down the stack, generating the elements
c     of lists 1, 3 for the box jbox, as appropriate
      do ijk=1,1 000 000 000
c
c     if this box is separated from ibox - store it in list 3;
c     enter this fact in the parent's table; pass control
c     to the parent
c
         call intersect(corners(1,1,ibox),corners(1,1,jbox),ifinter)
c
         if(ifinter .eq. 1) exit
         call setList(3,ibox,nboxes,jbox,1)
c
c     if storage capacity has been exceeed - bomb
c
         istack=istack-1
         stack(3,istack)=stack(3,istack)-1
         jbox=stack(2,istack)
      enddo
c
c     this box is not separated from ibox. if it is childless
c     - enter it in list 1; enter this fact in the parent's table;
c     pass control to the parent
c
      if(boxes(6,jbox) .eq. 0) then
         call setList(1,ibox,nboxes,jbox,1)
c     . . . entered jbox in the list1 of ibox; if jbox
c     is on the finer level than ibox - enter ibox
c     in the list 1 of jbox
         if(boxes(1,jbox) .ne. boxes(1,ibox)) then
            call setList(1,jbox,nboxes,ibox,1)
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
c
c
      subroutine intersect(c1,c2,ifinter)
      implicit real *8 (a-h,o-z)
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
      implicit real *8 (a-h,o-z)
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
c
      subroutine growTree(z,n,ncrit,boxes,maxboxes,
     1     nboxes,iz,nlev,center0,size,
     1     minlevel,maxlevel)
      use arrays, only : levelOffset
      implicit real *8 (a-h,o-z)
      integer boxes(20,*),iz(*),iwork(n),
     1     is(8),ns(8),
     1     iichilds(8),jjchilds(8),kkchilds(8)
      real *8 z(3,*),center0(3),center(3)
      data kkchilds/1,1,1,1,2,2,2,2/,jjchilds/1,1,2,2,1,1,2,2/,
     1     iichilds/1,2,1,2,1,2,1,2/
      xmin=z(1,1)
      xmax=z(1,1)
      ymin=z(2,1)
      ymax=z(2,1)
      zmin=z(3,1)
      zmax=z(3,1)
      do i=1,n
         if(z(1,i) .lt. xmin) xmin=z(1,i)
         if(z(1,i) .gt. xmax) xmax=z(1,i)
         if(z(2,i) .lt. ymin) ymin=z(2,i)
         if(z(2,i) .gt. ymax) ymax=z(2,i)
         if(z(3,i) .lt. zmin) zmin=z(3,i)
         if(z(3,i) .gt. zmax) zmax=z(3,i)
      enddo
      size=xmax-xmin
      sizey=ymax-ymin
      sizez=zmax-zmin
      if(sizey .gt. size) size=sizey
      if(sizez .gt. size) size=sizez
      center0(1)=(xmin+xmax)/2
      center0(2)=(ymin+ymax)/2
      center0(3)=(zmin+zmax)/2
      boxes(1,1)=0
      boxes(2,1)=1
      boxes(3,1)=1
      boxes(4,1)=1
      boxes(5,1)=0
      boxes(6,1)=0
      boxes(7,1)=0
      boxes(8,1)=0
      boxes(9,1)=0
      boxes(10,1)=0
      boxes(11,1)=0
      boxes(12,1)=0
      boxes(13,1)=0
      boxes(14,1)=1
      boxes(15,1)=n
      boxes(16,1)=1
      boxes(17,1)=0
      if( n .le. 0 ) boxes(18,1)=0
      if( n .gt. 0 ) boxes(18,1)=1
      boxes(19,1)=0
      boxes(20,1)=0
c
      levelOffset(1)=1
      levelOffset(2)=2
c
      do i=1,n
         iz(i)=i
      enddo
c     recursively (one level after another) subdivide all
c     boxes till none are left with more than ncrit particles
      maxChild=maxboxes
      maxlev=198
      if( maxlevel .lt. maxlev ) maxlev=maxlevel
      ichild=1
      nlev=0
      do level=0,maxlev-1
         nlevChild=0
         do iparent=levelOffset(level+1),levelOffset(level+2)-1
c     subdivide the box number iparent (if needed)
            nump=boxes(15,iparent)
            numt=boxes(17,iparent)
c     ... refine on both sources and targets
            if(nump.le.ncrit.and.numt.le.ncrit.and.
     $           level.ge.minlevel) cycle
c     ... not a leaf node on sources or targets
            if(nump.gt.ncrit.or.numt.gt.ncrit)then
               if(boxes(18,iparent).eq.1) boxes(18,iparent)=2
               if(boxes(19,iparent).eq.1) boxes(19,iparent)=2
            endif
            ii=boxes(2,iparent)
            jj=boxes(3,iparent)
            kk=boxes(4,iparent)
            call findCenter(center0,size,level,ii,jj,kk,center)
            iiz=boxes(14,iparent)
            nz=boxes(15,iparent)
            call reorder(center,z,iz(iiz),nz,iwork,is,ns)
            ic=6
            do i=1,8
               if(ns(i).eq.0) cycle
               nlevChild=nlevChild+1
               ichild=ichild+1
               nlev=level+1
               if(ichild.gt.maxChild) then
                  print*,"Error: ibox out of bounds"
                  stop
               endif
c     store in array boxes all information about this son
               do lll=6,13
                  boxes(lll,ichild)=0
               enddo
               boxes(1,ichild)=level+1
               iichild=(ii-1)*2+iichilds(i)
               jjson=(jj-1)*2+jjchilds(i)
               kkson=(kk-1)*2+kkchilds(i)
               boxes(2,ichild)=iichild
               boxes(3,ichild)=jjson
               boxes(4,ichild)=kkson
               boxes(5,ichild)=iparent
               boxes(14,ichild)=is(i)+iiz-1
               boxes(15,ichild)=ns(i)
               boxes(16,ichild)=0
               boxes(17,ichild)=0
               if( ns(i) .le. 0 ) boxes(18,ichild)=0
               if( ns(i) .gt. 0 ) boxes(18,ichild)=1
               boxes(19,ichild)=0
               boxes(20,ichild)=0
               boxes(ic,iparent)=ichild
               ic=ic+1
               nboxes=ichild
            enddo
         enddo
         levelOffset(level+3)=levelOffset(level+2)+nlevChild
         if(nlevChild .eq. 0) exit
         level1=level
      enddo
      if( level1 .ge. 197 ) then
         print*,"Error: level out of bounds"
         stop
      endif
      nboxes=ichild
      return
      end
c
c
c
c
c
      subroutine setCenter(center0,size,nboxes)
      use arrays, only : boxes,centers,corners
      implicit real *8 (a-h,o-z)
      real *8 center(3),center0(3)
ccc   save
c
c     this subroutine produces arrays of centers and
c     corners for all boxes in the oct-tree structure.
c
c     input parameters:
c
c     center0 - the center of the box on the level 0, containing
c     the whole simulation
c     size - the side of the box on the level 0
c     boxes - an integer array dimensioned (10,nboxes), as produced
c     by the subroutine d3tallb (see)
c     column describes one box, as follows:
c     nboxes - the total number of boxes created
c
c
c     output parameters:
c
c     centers - the centers of all boxes in the array boxes
c     corners - the corners of all boxes in the array boxes
c
c     . . . construct the corners for all boxes
c
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
         center(1)=x00+(ii-1)*side+side2
         center(2)=y00+(jj-1)*side+side2
         center(3)=z00+(kk-1)*side+side2
c
         centers(1,i)=center(1)
         centers(2,i)=center(2)
         centers(3,i)=center(3)
c
         corners(1,1,i)=center(1)-side/2
         corners(1,2,i)=corners(1,1,i)
         corners(1,3,i)=corners(1,1,i)
         corners(1,4,i)=corners(1,1,i)
         corners(1,5,i)=corners(1,1,i)+side
         corners(1,6,i)=corners(1,5,i)
         corners(1,7,i)=corners(1,5,i)
         corners(1,8,i)=corners(1,5,i)
c
         corners(2,1,i)=center(2)-side/2
         corners(2,2,i)=corners(2,1,i)
         corners(2,5,i)=corners(2,1,i)
         corners(2,6,i)=corners(2,1,i)
         corners(2,3,i)=corners(2,1,i)+side
         corners(2,4,i)=corners(2,3,i)
         corners(2,7,i)=corners(2,3,i)
         corners(2,8,i)=corners(2,3,i)
c
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
c
c
c
c
c
      subroutine findCenter(center0,size,level,i,j,k,center)
      implicit real *8 (a-h,o-z)
      real *8 center(3),center0(3)
      data level0/-1/
c     this subroutine finds the center of the box
c     number (i,j) on the level level. note that the
c     box on level 0 is assumed to have the center
c     center0, and the side size
      side=size/2**level
      side2=side/2
      x0=center0(1)-size/2
      y0=center0(2)-size/2
      z0=center0(3)-size/2
      level0=level
      center(1)=x0+(i-1)*side+side2
      center(2)=y0+(j-1)*side+side2
      center(3)=z0+(k-1)*side+side2
      return
      end

c
c
c
c
c
      subroutine reorder(cent,z,iz,n,iwork,
     1     is,ns)
      implicit real *8 (a-h,o-z)
      real *8 cent(3),z(3,*)
      integer iz(*),iwork(*),is(*),ns(*)
ccc   save
c
c     this subroutine reorders the particles in a box,
c     so that each of the children occupies a contigious
c     chunk of array iz
c
c     note that we are using a standard numbering convention
c     for the children:
c
c
c     5,6     7,8
c
c     <- looking down the z-axis
c     1,2     3,4
c
c
cccc  in the original code, we were using a strange numbering convention
cccc  for the children:
cccc
cccc  3,4     7,8
cccc
cccc  <- looking down the z-axis
cccc  1,2     5,6
c
c
c     input parameters:
c
c     cent - the center of the box to be subdivided
c     z - the list of all points in the box to be subdivided
c     iz - the integer array specifying the transposition already
c     applied to the points z, before the subdivision of
c     the box into children
c     n - the total number of points in array z
c
c     output parameters:
c
c     iz - the integer array specifying the transposition already
c     applied to the points z, after the subdivision of
c     the box into children
c     is - an integer array of length 8 containing the locations
c     of the childs in array iz
c     ns - an integer array of length 8 containig the numbers of
c     elements in the childs
c
c     work arrays:
c
c     iwork - must be n integer elements long
c
c     . . . separate all particles in this box in x
c
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
      itype=3
      thresh=cent(3)
      call divide(z,iz,n,itype,thresh,iwork,n1234)
      n5678=n-n1234
c     at this point, the contents of childs number 1,2,3,4 are in
c     the part of array iz with numbers 1,2,...n1234
c     the contents of childs number 5,6,7,8  are in
c     the part of array iz with numbers n1234+1,n12+2,...,n
c     . . . separate the boxes 1, 2, 3, 4 and boxes 5, 6, 7, 8
      itype=2
      thresh=cent(2)
      if(n1234 .ne. 0)
     1     call divide(z,iz,n1234,itype,thresh,iwork,n12)
      n34=n1234-n12
      if(n5678 .ne. 0)
     1     call divide(z,iz(n1234+1),n5678,itype,thresh,iwork,n56)
      n78=n5678-n56
c     perform the final separation of pairs of sonnies into
c     individual ones
      itype=1
      thresh=cent(1)
      if(n12 .ne. 0)
     1     call divide(z,iz,n12,itype,thresh,iwork,n1)
      n2=n12-n1
      if(n34 .ne. 0)
     1     call divide(z,iz(n12+1),n34,itype,thresh,iwork,n3)
      n4=n34-n3
      if(n56 .ne. 0)
     1     call divide(z,iz(n1234+1),n56,itype,thresh,iwork,n5)
      n6=n56-n5
      if(n78 .ne. 0)
     1     call divide(z,iz(n1234+n56+1),n78,itype,thresh,iwork,n7)
      n8=n78-n7
c     store the information about the sonnies in appropriate arrays
      is(1)=1
      ns(1)=n1
      is(2)=is(1)+ns(1)
      ns(2)=n2
      is(3)=is(2)+ns(2)
      ns(3)=n3
      is(4)=is(3)+ns(3)
      ns(4)=n4
      is(5)=is(4)+ns(4)
      ns(5)=n5
      is(6)=is(5)+ns(5)
      ns(6)=n6
      is(7)=is(6)+ns(6)
      ns(7)=n7
      is(8)=is(7)+ns(7)
      ns(8)=n8
      return
      end

      subroutine divide(z,iz,n,itype,thresh,iwork,n1)
      implicit none
      integer i1,i2,i,j,n1,n,itype
      integer iz(*),iwork(*)
      real *8 thresh
      real *8 z(3,*)
      i1=0
      i2=0
      do i=1,n
         j=iz(i)
         if(z(itype,j).le.thresh) then
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

      subroutine setList(itype,ibox,nboxes,list,nlist)
      use arrays, only : listOffset,lists
      implicit none
      integer ilast,ibox,itype,nlist,i,numele,nboxes
      integer list(*)
      data numele/0/
      ilast=listOffset(ibox,itype)
      do i=1,nlist
         numele=numele+1
         lists(1,numele)=ilast
         lists(2,numele)=list(i)
         ilast=numele
      enddo
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
