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
      call setLists(nboxes)
      return
      end

      subroutine setLists(nboxes)
      use arrays, only : boxes,listOffset,corners
      implicit none
      integer i,j,k,ibox,jbox,nboxes,iparent,nkids,icoll,ncolls,kid
      integer nlist
      integer kids(50000),parents(2000),list5(20000)
      do k=1,5
         do i=1,nboxes
            listOffset(i,k)=-1
         enddo
      enddo
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
               if (kid.gt.0) then
                  if (kid.ne.ibox) then
                     nkids=nkids+1
                     kids(nkids)=kid
                  endif
               endif
            enddo
         enddo
         do i=1,nkids
            jbox=kids(i)
            if ( boxes(2,ibox)-1.le.boxes(2,jbox).and.
     1           boxes(2,ibox)+1.ge.boxes(2,jbox).and.
     1           boxes(3,ibox)-1.le.boxes(3,jbox).and.
     1           boxes(3,ibox)+1.ge.boxes(3,jbox).and.
     1           boxes(4,ibox)-1.le.boxes(4,jbox).and.
     1           boxes(4,ibox)+1.ge.boxes(4,jbox) ) then
               call setList(5,ibox,jbox)
            else
               call setList(2,ibox,jbox)
            endif
         enddo
      enddo
      do ibox=1,nboxes
         if (boxes(6,ibox).eq.0) then
            call getList(5,ibox,list5,nlist)
            do j=1,nlist
               jbox=list5(j)
               if (boxes(6,jbox).eq.0) then
                  call setList(1,ibox,jbox)
                  if (boxes(1,jbox).ne.boxes(1,ibox)) then
                     call setList(1,jbox,ibox)
                  endif
               endif
            enddo
         endif
      enddo
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
     1     nboxes,permutation,numLevels,X0,size)
      use arrays, only : levelOffset
      implicit none
      integer i,numLevels,level
      integer iparent,nchild,ibody,nbody,ncrit,numBodies
      integer offset,nboxes
      integer boxes(20,*),permutation(*),iwork(numBodies),nbody8(8)
      real *8 xmin,xmax,ymin,ymax,zmin,zmax
      real *8 size,sizey,sizez,R0
      real *8 Xj(3,*),X0(3)
      xmin=Xj(1,1)
      xmax=Xj(1,1)
      ymin=Xj(2,1)
      ymax=Xj(2,1)
      zmin=Xj(3,1)
      zmax=Xj(3,1)
      do i=1,numBodies
         if (Xj(1,i).lt.xmin) xmin=Xj(1,i)
         if (Xj(1,i).gt.xmax) xmax=Xj(1,i)
         if (Xj(2,i).lt.ymin) ymin=Xj(2,i)
         if (Xj(2,i).gt.ymax) ymax=Xj(2,i)
         if (Xj(3,i).lt.zmin) zmin=Xj(3,i)
         if (Xj(3,i).gt.zmax) zmax=Xj(3,i)
      enddo
      size=xmax-xmin
      sizey=ymax-ymin
      sizez=zmax-zmin
      if (sizey.gt.size) size=sizey
      if (sizez.gt.size) size=sizez
      R0=size/2
      X0(1)=(xmin+xmax)/2
      X0(2)=(ymin+ymax)/2
      X0(3)=(zmin+zmax)/2
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
      numLevels=0
      do level=1,198
         do iparent=levelOffset(level),levelOffset(level+1)-1
            nbody=boxes(9,iparent)
            if (nbody.le.ncrit) cycle
            ibody=boxes(8,iparent)
            call reorder(X0,R0,level,boxes(2,iparent),
     1           Xj,permutation(ibody),nbody,iwork,nbody8)
            nchild=0
            offset=ibody
            boxes(6,iparent)=nboxes+1
            do i=0,7
               if (nbody8(i+1).eq.0) cycle
               nboxes=nboxes+1
               numLevels=level
               boxes(1,nboxes)=level
               boxes(2,nboxes)=boxes(2,iparent)*2+mod(i,2)
               boxes(3,nboxes)=boxes(3,iparent)*2+mod(i/2,2)
               boxes(4,nboxes)=boxes(4,iparent)*2+i/4
               boxes(5,nboxes)=iparent
               boxes(6,nboxes)=0
               boxes(7,nboxes)=0
               boxes(8,nboxes)=offset
               boxes(9,nboxes)=nbody8(i+1)
               nchild=nchild+1
               offset=offset+nbody8(i+1)
            enddo
            boxes(7,iparent)=nchild
         enddo
         levelOffset(level+2)=nboxes+1
         if (levelOffset(level+1).eq.levelOffset(level+2)) exit
      enddo
      return
      end

      subroutine setCenter(X0,size,nboxes)
      use arrays, only : boxes,centers,corners
      implicit none
      integer i,nboxes,level
      real *8 x00,y00,z00,side,side2,size
      real *8 center(3),X0(3)
      x00=X0(1)-size/2
      y00=X0(2)-size/2
      z00=X0(3)-size/2
      do i=1,nboxes
         level=boxes(1,i)
         side=size/2**level
         side2=side/2
         center(1)=x00+boxes(2,i)*side+side2
         center(2)=y00+boxes(3,i)*side+side2
         center(3)=z00+boxes(4,i)*side+side2
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

      subroutine reorder(X0,R0,level,iX,
     1     Xj,index,n,iwork,nbody)
      implicit none
      integer n,d,i,j,level,octant
      integer iX(3),offset(9)
      integer index(*),iwork(*),nbody(*)
      real *8 R,R0
      real *8 X(3),X0(3),Xj(3,*)
      R=R0/2**(level-1)
      do d=1,3
         X(d)=X0(d)-R0+iX(d)*R*2+R
      enddo
      do i=1,8
         nbody(i)=0
      enddo
      do i=1,n
         j=index(i)
         octant=-(Xj(3,j).gt.X(3))*4-(Xj(2,j).gt.X(2))*2
     1        -(Xj(1,j).gt.X(1))+1
         nbody(octant)=nbody(octant)+1
      enddo
      offset(1)=1
      do i=1,8
         offset(i+1)=offset(i)+nbody(i)
         nbody(i)=0
      enddo
      do i=1,n
         j=index(i)
         octant=-(Xj(3,j).gt.X(3))*4-(Xj(2,j).gt.X(2))*2
     1        -(Xj(1,j).gt.X(1))+1
         iwork(offset(octant)+nbody(octant))=index(i)
         nbody(octant)=nbody(octant)+1
      enddo
      do i=1,n
         index(i)=iwork(i)
      enddo
      return
      end

      subroutine setList(itype,ibox,list)
      use arrays, only : listOffset,lists
      implicit none
      integer ibox,itype,list,numele/0/
      numele=numele+1
      lists(1,numele)=listOffset(ibox,itype)
      lists(2,numele)=list
      listOffset(ibox,itype)=numele
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
         if (lists(2,ilast).gt.0) then
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
