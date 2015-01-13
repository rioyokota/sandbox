      subroutine getBounds(Xj,numBodies,X0,R0)
      integer numBodies,i,d
      real *8 R0,diameter/0.0/
      real *8 Xj(3,*),Xmin(3),Xmax(3),X0(3)
      do d=1,3
         Xmin(d)=Xj(d,1)
         Xmax(d)=Xj(d,1)
      enddo
      do i=1,numBodies
         do d=1,3
            if (Xj(d,i).lt.Xmin(d)) Xmin(d)=Xj(d,i)
            if (Xj(d,i).gt.Xmax(d)) Xmax(d)=Xj(d,i)
         enddo
      enddo
      do d=1,3
         if (Xmax(d)-Xmin(d).gt.diameter) diameter=Xmax(d)-Xmin(d)
         X0(d)=(Xmax(d)+Xmin(d))*0.5
      enddo
      R0=diameter*0.5
      return
      end

      subroutine buildTree(Xj,numBodies,ncrit,
     1     nboxes,permutation,nlev,X0,R0)
      use arrays, only : listOffset,lists,nodes,boxes,centers
      implicit none
      integer i,j,d,numBodies,ncrit,nboxes,nlev
      integer permutation(*)
      real *8 R,R0
      real *8 Xj(3,*),X0(3)
      do i=1,numBodies
         permutation(i)=i
      enddo
      allocate(nodes(20,numBodies))
      call growTree(Xj,numBodies,ncrit,nodes,
     1     nboxes,permutation,nlev,X0,R0)
      allocate(listOffset(nboxes,5))
      allocate(lists(2,189*nboxes))
      allocate(boxes(20,nboxes))
      allocate(centers(3,nboxes))
      do i=1,nboxes
         do j=1,20
            boxes(j,i)=nodes(j,i)
         enddo
      enddo
      do i=1,nboxes
         R=R0/2**boxes(1,i)
         do d=1,3
            centers(d,i)=X0(d)-R0+boxes(d+1,i)*R*2+R
         enddo
      enddo
      call setLists(nboxes)
      return
      end

      subroutine growTree(Xj,numBodies,ncrit,boxes,
     1     nboxes,permutation,numLevels,X0,R0)
      use arrays, only : levelOffset
      implicit none
      integer i,numLevels,level
      integer iparent,nchild,ibody,nbody,ncrit,numBodies
      integer offset,nboxes
      integer boxes(20,*),permutation(*),iwork(numBodies),nbody8(8)
      real *8 R0
      real *8 Xj(3,*),X0(3)
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

      subroutine reorder(X0,R0,level,iX,
     1     Xj,permutation,n,iwork,nbody)
      implicit none
      integer n,d,i,j,level,octant
      integer iX(3),offset(9)
      integer permutation(*),iwork(*),nbody(*)
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
         j=permutation(i)
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
         j=permutation(i)
         octant=-(Xj(3,j).gt.X(3))*4-(Xj(2,j).gt.X(2))*2
     1        -(Xj(1,j).gt.X(1))+1
         iwork(offset(octant)+nbody(octant))=permutation(i)
         nbody(octant)=nbody(octant)+1
      enddo
      do i=1,n
         permutation(i)=iwork(i)
      enddo
      return
      end

      subroutine setLists(nboxes)
      use arrays, only : boxes,listOffset
      implicit none
      integer i,j,ibox,jbox,nboxes,nchilds,numNeighbors
      integer iparent,jparent
      integer childs(216),neighbors(27)
      do j=1,5
         do i=1,nboxes
            listOffset(i,j)=-1
         enddo
      enddo
      do ibox=2,nboxes
         iparent=boxes(5,ibox)
         neighbors(1)=iparent
         call getList(5,iparent,neighbors(2),numNeighbors)
         numNeighbors=numNeighbors+1
         nchilds=0
         do i=1,numNeighbors
            jparent=neighbors(i)
            do j=1,boxes(7,jparent)
               jbox=boxes(6,jparent)+j-1
               if (jbox.ne.ibox) then
                  nchilds=nchilds+1
                  childs(nchilds)=jbox
               endif
            enddo
         enddo
         do i=1,nchilds
            jbox=childs(i)
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
            call getList(5,ibox,neighbors,numNeighbors)
            do j=1,numNeighbors
               jbox=neighbors(j)
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
      integer ilast,ibox,itype,nlist
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
      return
      end
