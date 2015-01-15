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

      subroutine buildTree(Xj,numBodies,
     1     numCells,permutation,numLevels,X0,R0)
      use arrays, only : listOffset,lists,nodes,cells,centers
      implicit none
      integer i,j,d,numBodies,numCells,numLevels
      integer permutation(*)
      real *8 R,R0
      real *8 Xj(3,*),X0(3)
      do i=1,numBodies
         permutation(i)=i
      enddo
      allocate(nodes(10,numBodies))
      call growTree(Xj,numBodies,nodes,
     1     numCells,permutation,numLevels,X0,R0)
      allocate(listOffset(numCells,3))
      allocate(lists(2,189*numCells))
      allocate(cells(10,numCells))
      allocate(centers(3,numCells))
      do i=1,numCells
         do j=1,10
            cells(j,i)=nodes(j,i)
         enddo
      enddo
      do i=1,numCells
         R=R0/2**cells(1,i)
         do d=1,3
            centers(d,i)=X0(d)-R0+cells(d+1,i)*R*2+R
         enddo
      enddo
      call setLists(numCells)
      return
      end

      subroutine growTree(Xj,numBodies,cells,
     1     numCells,permutation,numLevels,X0,R0)
      use constants, only : P
      use arrays, only : levelOffset
      implicit none
      integer i,numLevels,level,ncrit
      integer iparent,nchild,ibody,nbody,numBodies
      integer offset,numCells
      integer cells(10,*),permutation(*),iwork(numBodies),nbody8(8)
      real *8 R0
      real *8 Xj(3,*),X0(3)
      cells(1,1)=0 ! level
      cells(2,1)=0 ! iX(1)
      cells(3,1)=0 ! iX(2)
      cells(4,1)=0 ! iX(3)
      cells(5,1)=0 ! iparent
      cells(6,1)=0 ! ichild
      cells(7,1)=0 ! nchild
      cells(8,1)=1 ! ibody
      cells(9,1)=numBodies ! nbody
      levelOffset(1)=1
      levelOffset(2)=2
      do i=1,numBodies
         permutation(i)=i
      enddo
      ncrit=1000
      if(P.lt.40) ncrit=500
      if(P.lt.30) ncrit=200
      if(P.lt.20) ncrit=100
      numCells=1
      numLevels=0
      do level=1,198
         do iparent=levelOffset(level),levelOffset(level+1)-1
            nbody=cells(9,iparent)
            if (nbody.le.ncrit) cycle
            ibody=cells(8,iparent)
            call reorder(X0,R0,level,cells(2,iparent),
     1           Xj,permutation(ibody),nbody,iwork,nbody8)
            nchild=0
            offset=ibody
            cells(6,iparent)=numCells+1
            do i=0,7
               if (nbody8(i+1).eq.0) cycle
               numCells=numCells+1
               numLevels=level
               cells(1,numCells)=level
               cells(2,numCells)=cells(2,iparent)*2+mod(i,2)
               cells(3,numCells)=cells(3,iparent)*2+mod(i/2,2)
               cells(4,numCells)=cells(4,iparent)*2+i/4
               cells(5,numCells)=iparent
               cells(6,numCells)=0
               cells(7,numCells)=0
               cells(8,numCells)=offset
               cells(9,numCells)=nbody8(i+1)
               nchild=nchild+1
               offset=offset+nbody8(i+1)
            enddo
            cells(7,iparent)=nchild
         enddo
         levelOffset(level+2)=numCells+1
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

      subroutine setLists(numCells)
      use arrays, only : cells,listOffset
      implicit none
      integer i,j,icell,jcell,numCells,nchilds,numNeighbors
      integer iparent,jparent
      integer childs(216),neighbors(27)
      do j=1,3
         do i=1,numCells
            listOffset(i,j)=-1
         enddo
      enddo
      do icell=2,numCells
         iparent=cells(5,icell)
         neighbors(1)=iparent
         call getList(3,iparent,neighbors(2),numNeighbors)
         numNeighbors=numNeighbors+1
         nchilds=0
         do i=1,numNeighbors
            jparent=neighbors(i)
            do j=1,cells(7,jparent)
               jcell=cells(6,jparent)+j-1
               if (jcell.ne.icell) then
                  nchilds=nchilds+1
                  childs(nchilds)=jcell
               endif
            enddo
         enddo
         do i=1,nchilds
            jcell=childs(i)
            if ( cells(2,icell)-1.le.cells(2,jcell).and.
     1           cells(2,icell)+1.ge.cells(2,jcell).and.
     1           cells(3,icell)-1.le.cells(3,jcell).and.
     1           cells(3,icell)+1.ge.cells(3,jcell).and.
     1           cells(4,icell)-1.le.cells(4,jcell).and.
     1           cells(4,icell)+1.ge.cells(4,jcell) ) then
               call setList(3,icell,jcell)
            else
               call setList(2,icell,jcell)
            endif
         enddo
      enddo
      do icell=1,numCells
         if (cells(6,icell).eq.0) then
            call getList(3,icell,neighbors,numNeighbors)
            do j=1,numNeighbors
               jcell=neighbors(j)
               if (cells(6,jcell).eq.0) then
                  call setList(1,icell,jcell)
                  if (cells(1,jcell).ne.cells(1,icell)) then
                     call setList(1,jcell,icell)
                  endif
               endif
            enddo
         endif
      enddo
      return
      end

      subroutine setList(itype,icell,list)
      use arrays, only : listOffset,lists
      implicit none
      integer icell,itype,list,numele/0/
      numele=numele+1
      lists(1,numele)=listOffset(icell,itype)
      lists(2,numele)=list
      listOffset(icell,itype)=numele
      return
      end

      subroutine getList(itype,icell,list,nlist)
      use arrays, only : listOffset,lists
      implicit none
      integer ilast,icell,itype,nlist
      integer list(*)
      ilast=listOffset(icell,itype)
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
