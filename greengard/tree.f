      subroutine buildTree(ier,z,n,ncrit,
     $     nboxes,iz,laddr,nlev,center,size,
     $     wlists,lwlists)
      use arrays, only : listOffset,lists
      implicit real *8 (a-h,o-z)
      integer iz(*),wlists(*),laddr(2,*)
      real *8 z(3,*),center(3),corners(3,8)
c        this subroutine constructs the logical structure for the 
c        fully adaptive FMM in three dimensions and stores it in the
c        array wlists in the form of a link-list. after that, the user 
c        can obtain the information about various boxes and lists 
c        in it by calling the entries getCell, getList, getMemory
c        of this subroutine (see).
c
c        this subroutine constructs the tree on sources and targets
c
c              note on the list conventions. 
c
c    list 1 of the box ibox - the list of all boxes with which the
c           box ibox interacts directly, including the boxes on the 
c           same level as ibox, boxes on the finer levels, and boxes 
c           on the coarser levels. obviously, list 1 is empty for any
c           box that is not childless.
c
c    list 2 of the box ibox - the list of all boxes with which the
c           box ibox interacts in the regular multipole fashion, i.e. 
c           boxes on the same level as ibox that are separated from it
c           but whose daddies are not separated from the daddy of ibox.
c
c    list 3 of the box ibox - for a childless ibox, the list of all 
c           boxes on the levels finer than that of ibox, which are 
c           separated from ibox, and whose daddys are not separated 
c           from ibox. for a box with children, list 3 is empty.
c           
c    list 4 is dual to list 3, i.e. jbox is on the list 4 of ibox if 
c           and only if ibox is on the list 3 of jbox. 
c
c    list 5 of the box ibox - the list of all boxes at the same level 
c           that are adjacent to the box ibox - the list of colleagues
c
c                            input parameters:
c
c  z - the user-specified points in the space
c  n - the number of elements in array z
c  ncrit - the maximum number of points in a box on the finest level
c
c                            output parameters:
c
c  laddr - an integer array dimensioned (2,nlev), describing the
c         numbers of boxes on various levels of sybdivision, so that
c         the first box on level (i-1) has sequence number laddr(1,i),
c         and there are laddr(2,i) boxes on level i-1
c  nlev - the maximum level number on which any boxes have 
c         been created. the maximum number possible is 200. 
c         it is recommended that the array laddr above be 
c         dimensioned at least (2,200), in case the user underestimates
c         the number of levels required.
c  center - the center of the box on the level 0, containing
c         the whole simulation
c  size - the side of the box on the level 0
        ier=0
        iiwork=501
        iboxes=iiwork+n+4
        lboxes=lwlists-n-9
        maxboxes=lboxes/20-1
c     initialize the sorting index 
        do i=1,n
           iz(i)=i
	enddo
        ifempty=0
        minlevel=0
        maxlevel=100
        call growTree(ier,z,n,ncrit,wlists(iboxes),maxboxes,
     1    nboxes,iz,laddr,nlev,center,size,wlists(iiwork),
     $     ifempty,minlevel,maxlevel)
c     compress the array wlists
        nn=nboxes*20
        do i=1,nn
           wlists(iiwork+i-1)=wlists(iboxes+i-1)
        enddo
        iboxes=iiwork
        lboxes=nboxes*20+20
c     align array for real *16 storage
        lboxes=lboxes+4-mod(lboxes,4)
c     construct the centers and the corners for all boxes in the oct-tree
        icenters=iboxes+lboxes
        lcenters=(nboxes*3+2)*2
        icorners=icenters+lcenters
        lcorners=(nboxes*24+2)*2
        iwlists=icorners+lcorners
        lwlists=lwlists-iwlists-6
        allocate(listOffset(nboxes,5))
        allocate(lists(2,189*nboxes))
        call setCenter(center,size,wlists(iboxes),nboxes,
     1      wlists(icenters),wlists(icorners) )
c     now, construct all lists for all boxes
        call getLists(ier,wlists(iboxes),nboxes,wlists(icorners),
     1        lwlists)
c     store all pointers
        wlists(1)=nboxes
        wlists(2)=iboxes
        wlists(3)=icorners
        wlists(4)=icenters
        wlists(5)=iwlists
        wlists(6)=0
        wlists(7)=n
        wlists(8)=ncrit
        wlists(9)=nlev
        wlists(10)=ier
        wlists(11)=0
        wlists(12)=ifempty
        wlists(13)=minlevel
        wlists(14)=maxlevel
        do i=1,200
           wlists(100+2*i-2)=laddr(1,i)
           wlists(100+2*i-1)=laddr(2,i)
        enddo
        return
        end

        subroutine getNumChild(box,nkids)
        implicit real *8 (a-h,o-z)
        integer box(20)
        nkids=0
        do ikid=1,8
           if( box(5+ikid) .ne. 0 ) nkids=nkids+1
        enddo
        return
        end

        subroutine getCell(ier,ibox,box,center,corners,w)
        use arrays, only : listOffset
        implicit real *8 (a-h,o-z)
        integer w(*),box(20)
        real *8 center(3),corners(3,8)
        data nboxes/0/
c
c        this entry returns to the user the characteristics of
c        user-specified box  ibox.  
c
c                     input parameters:
c
c  ibox - the box number for which the information is desired
c  w - storage area as created by the entry buildTree (see above)
c
c                     output parameters:
c
c  ier - the error return code.
c    ier=0 means successful execution
c    ier=2 means that ibox is either greater than the number of boxes
c           in the structure or less than 1.
c  box - an integer array dimensioned box(20). its elements describe 
c        the box number ibox, as follows:
c
c       1. level - the level of subdivision on which this box 
c             was constructed; 
c       2, 3, 4  - the coordinates of this box among  all
c             boxes on this level
c       5 - the daddy of this box, identified by it address
c             in array boxes
c       6,7,8,9,10,11,12,13 - the  list of children of this box 
c             (eight of them, and the child is identified by its address
c             in the array boxes; if a box has only one child, only the
c             first of the four child entries is non-zero, etc.)
c       14 - the location in the array iz of the particles 
c             living in this box
c       15 - the number of particles living in this box
c       16 - the location in the array iztarg of the targets
c             living in this box
c       17 - the number of targets living in this box
c       18 - source box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       19 - target box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       20 - reserved for future use
c
c  center - the center of the box number ibox 
c  corners - the corners of the box number ibox 
c
c       . . . return to the user all information about the box ibox
c 
        nboxes=w(1)
        iboxes=w(2)
        icorners=w(3)
        icenters=w(4)
        iwlists=w(5)
c 
        ier=0
        if( (ibox .ge. 1)  .and. (ibox .le. nboxes) ) goto 2100
        ier=2
        return
 2100 continue
c
        ibox0=iboxes+(ibox-1)*20-1
        do 2200 i=1,20
        box(i)=w(ibox0+i)
 2200 continue
c
c      return to the user the center and the corners of the box ibox
c
        call getCenter(w(icenters),w(icorners),ibox,center,corners) 
        return
c
        entry getList(ier,ibox,itype,list,nlist)
        call d3tlinkretr(ier,itype,ibox,nboxes,list,nlist)
        return
        end
c
        subroutine getLists(ier,boxes,nboxes,corners,lw)
        use arrays, only : listOffset
        implicit real *8 (a-h,o-z)
        integer boxes(20,*),collkids(50000),
     1      dadcolls(2000),list5(20000),stack(60000)
        real *8 corners(3,8,*)
ccc        save
c
c        this subroutine constructs all lists for all boxes 
c        and stores them in the storage area w in the form
c        of a link list. the resulting data can be accessed 
c        by calls to various entries of the subroutine d3tlinkinit (see).
c
c                          input parameters:
c
c  boxes - an integer array dimensioned (20,nboxes), as created by 
c        the subroutine d3tallb (see).  each 20-element column
c         describes one box, as follows:
c
c
c       1. level - the level of subdivision on which this box 
c             was constructed; 
c       2, 3, 4  - the coordinates of this box among  all
c             boxes on this level
c       5 - the daddy of this box, identified by it address
c             in array boxes
c       6,7,8,9,10,11,12,13 - the  list of children of this box 
c             (eight of them, and the child is identified by its address
c             in the array boxes; if a box has only one child, only the
c             first of the four child entries is non-zero, etc.)
c       14 - the location in the array iz of the particles 
c             living in this box
c       15 - the number of particles living in this box
c       16 - the location in the array iztarg of the targets
c             living in this box
c       17 - the number of targets living in this box
c       18 - source box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       19 - target box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       20 - reserved for future use
c
c  nboxes - the total number of boxes created
c  corners - the array of corners of all the boxes to in array boxes
c  lw - the total amount of storage in array w (in integer words)
c 
c              output parameters:
c
c  ier - the error return code.
c    ier=0 means successful execution
c  w - storage area containing all lists for all boxes in 
c        the form of link-lists, accessible by the subroutine 
c        d3tlinkretr (see).
        ier=0
        ntypes=5
        call d3tlinkinit(jer,nboxes,ntypes,lw)
c
c        construct lists 5,2 for all boxes
c
        do 2000 ibox=2,nboxes
c
c       find this guy's daddy
c
        idad=boxes(5,ibox)
c
c       find daddy's collegues, including daddy himself
c
        dadcolls(1)=idad
        itype5=5
        itype2=2
        call d3tlinkretr(jer,itype5,idad,nboxes,dadcolls(2),ncolls)
        ncolls=ncolls+1
c
c        find the children of the daddy's collegues
c
        nkids=0
        do 1600 i=1,ncolls
        icoll=dadcolls(i)
        do 1400 j=1,8
        kid=boxes(5+j,icoll)
        if(kid .le. 0) goto 1600
        if(kid .eq. ibox) goto 1400
        nkids=nkids+1
        collkids(nkids)=kid
 1400 continue
 1600 continue
c
c       sort the kids of the daddy's collegues into the 
c       lists 2, 5 of the box ibox
c
        nlist1=1
        do 1800 i=1,nkids
c
c       check if this kid is touching the box ibox
c
        kid=collkids(i)
        call d3tifint(corners(1,1,kid),corners(1,1,ibox),ifinter)
ccc        call d3tifint2(boxes(1,kid),boxes(1,ibox),ifinter)
c
        if(ifinter .eq. 1)
     $    call d3tlinkstor(ier,itype5,ibox,nboxes,kid,nlist1)
c
c        if storage capacity has been exceeed - bomb
c
        if(ifinter .eq. 0)
     $    call d3tlinkstor(ier,itype2,ibox,nboxes,kid,nlist1)
 1800 continue
c
c        if storage capacity has been exceeed - bomb
c
 2000 continue
c
c
c       now, construct lists 1, 3
c
        do 3000 i=1,nboxes
c
c       if this box has kids - its lists 1, 3 are empty;
c       do not construct them
c
        if(boxes(6,i) .gt. 0) goto 3000
c
c       do not construct lists 1, 3 for the main box
c
        if(boxes(1,i) .eq. 0) goto 3000
c
        call d3tlinkretr(jer,itype5,i,nboxes,list5,nlist)  
c
        if(jer .eq. 4) goto 3000
c
        do 2200 j=1,nlist
        jbox=list5(j)
        call d3tlst31(ier,i,jbox,boxes,nboxes,
     1    corners,stack)
c
c        if storage capacity has been exceeded - bomb
c
 2200 continue
 3000 continue
c
c
        itype3=3
        itype4=4
        nlist1=1
        do 4400 ibox=1,nboxes
c
        call d3tlinkretr(jer,itype3,ibox,nboxes,list5,nlist)
        if(jer .eq. 4) goto 4400
        do 4200 j=1,nlist
        call d3tlinkstor(ier,itype4,list5(j),nboxes,ibox,nlist1)
c
c        if storage capacity has been exceeed - bomb
c
 4200 continue
 4400 continue

        return
        end        
c
c
c
c
c
        subroutine d3tlst31(ier,ibox,jbox0,boxes,nboxes,
     1    corners,stack)
        implicit real *8 (a-h,o-z)
        real *8 corners(3,8,*)
        integer boxes(20,*),stack(3,*)
        data itype1/1/,itype2/2/,itype3/3/,itype4/4/,itype5/5/,
     1      nlist1/1/
ccc        save
c
c       this subroutine constructs all elements of lists 1 and 3 
c       resulting from the subdivision of one element of list 5
c       of the box ibox. all these elements of lists 1, 3 are 
c       stored in the link-lists by the subroutine linstro (see)
c
c        input parameters:
c
c  ibox - the box whose lists are being constructed
c  
c  jbox0 - the element of list 5 of the box ibox that is being
c          subdfivided
c  boxes - the array boxes as created by the subroutine d3tallb (see)
c  nboxes - the number of boxes in array boxes
c  corners - the array of corners of all the boxes to in array boxes
c  w - the storage area formatted by the subroutine d3tlinkinit (see) 
c          to be used to store the elements of lists 1, 3 constructed
c          by this subroutine. obviously, by this time, it contains
c          plenty of other lists.
c  
c                      output parameters:
c
c  ier - the error return code.
c    ier=0 means successful execution
c  w - the augmented storage area, containing all the boxes just 
c          created, in addition to whatever had been stored previously
c
c                      work arrays:
c
c  stack - must be at least 600 integer locations long
c
c       . . . starting with the initial element of list 5, 
c             subdivide the boxes recursively and store 
c             the pieces where they belong
c
c        . . . initialize the process
c
        jbox=jbox0
        istack=1
        stack(1,1)=1
        stack(2,1)=jbox
c
        nsons=0
        do 1200 j=6,13
c
        if(boxes(j,jbox) .gt. 0) nsons=nsons+1
 1200 continue
c
        stack(3,1)=nsons
c
c       . . . move up and down the stack, generating the elements 
c             of lists 1, 3 for the box jbox, as appropriate
c
        do 5000 ijk=1,1 000 000 000
c
c       if this box is separated from ibox - store it in list 3;
c       enter this fact in the daddy's table; pass control
c       to the daddy
c
        call d3tifint(corners(1,1,ibox),corners(1,1,jbox),ifinter)
c
        if(ifinter .eq. 1) goto 2000
        call d3tlinkstor(ier,itype3,ibox,nboxes,jbox,nlist1)
c
c        if storage capacity has been exceeed - bomb
c
        istack=istack-1
        stack(3,istack)=stack(3,istack)-1
        jbox=stack(2,istack)
        goto 5000
 2000 continue
c
c       this box is not separated from ibox. if it is childless 
c       - enter it in list 1; enter this fact in the daddy's table; 
c       pass control to the daddy
c       
        if(boxes(6,jbox) .ne. 0) goto 3000
        call d3tlinkstor(ier,itype1,ibox,nboxes,jbox,nlist1)
c
c        if storage capacity has been exceeed - bomb
c
c       . . . entered jbox in the list1 of ibox; if jbox
c             is on the finer level than ibox - enter ibox
c             in the list 1 of jbox
c
        if(boxes(1,jbox) .eq. boxes(1,ibox)) goto 2400
        call d3tlinkstor(ier,itype1,jbox,nboxes,ibox,nlist1)
c
c        if storage capacity has been exceeed - bomb
c
 2400 continue
c
c       if we have processed the whole box jbox0, get out
c       of the subroutine
c
        if(jbox .eq. jbox0) return
c
        istack=istack-1
        stack(3,istack)=stack(3,istack)-1
        jbox=stack(2,istack)
        goto 5000
 3000 continue
c
c       this box is not separated from ibox, and has children. if 
c       the number of unprocessed sons of this box is zero 
c       - pass control to his daddy
c
        nsons=stack(3,istack)
        if(nsons .ne. 0) goto 4000
c
        if(jbox .eq. jbox0) return
c
        istack=istack-1
        stack(3,istack)=stack(3,istack)-1
        jbox=stack(2,istack)
        goto 5000
 4000 continue
c
c       this box is not separated from ibox; it has sons, and
c       not all of them have been processed. construct the stack
c       element for the appropriate son, and pass the control
c       to him. 
c
        jbox=boxes(5+nsons,jbox)
        istack=istack+1
c
        nsons=0
        do 4600 j=6,13
c
        if(boxes(j,jbox) .gt. 0) nsons=nsons+1
 4600 continue
c
        stack(1,istack)=istack
        stack(2,istack)=jbox
        stack(3,istack)=nsons
c
 5000 continue
c
        return
        end
c
c
c
c
c
        subroutine d3tifint(c1,c2,ifinter)
        implicit real *8 (a-h,o-z)
        real *8 c1(3,8),c2(3,8)
ccc        save
c
c        this subroutine determines if two boxes in the square 
c        intersect or touch.
c
c                input parameters:
c
c  c1 - the four corners of the first box
c  c2 - the four corners of the second box
c
c                output parametes:
c 
c  ifinter - the indicator.
c      ifinter=1 means that the boxes intersect
c      ifinter=0 means that the boxes do not intersect
c
c       . . . find the maximum and minimum coordinates
c             for both boxes
c
        xmin1=c1(1,1)
        ymin1=c1(2,1)
        zmin1=c1(3,1)
c
        xmax1=c1(1,1)
        ymax1=c1(2,1)
        zmax1=c1(3,1)
c
        xmin2=c2(1,1)
        ymin2=c2(2,1)
        zmin2=c2(3,1)
c
        xmax2=c2(1,1)
        ymax2=c2(2,1)
        zmax2=c2(3,1)
c
        do 1200 i=1,8
c
        if(xmin1 .gt. c1(1,i)) xmin1=c1(1,i)
        if(ymin1 .gt. c1(2,i)) ymin1=c1(2,i)
        if(zmin1 .gt. c1(3,i)) zmin1=c1(3,i)
c
        if(xmax1 .lt. c1(1,i)) xmax1=c1(1,i)
        if(ymax1 .lt. c1(2,i)) ymax1=c1(2,i)
        if(zmax1 .lt. c1(3,i)) zmax1=c1(3,i)
c
c
        if(xmin2 .gt. c2(1,i)) xmin2=c2(1,i)
        if(ymin2 .gt. c2(2,i)) ymin2=c2(2,i)
        if(zmin2 .gt. c2(3,i)) zmin2=c2(3,i)
c
        if(xmax2 .lt. c2(1,i)) xmax2=c2(1,i)
        if(ymax2 .lt. c2(2,i)) ymax2=c2(2,i)
        if(zmax2 .lt. c2(3,i)) zmax2=c2(3,i)
c
 1200 continue
c        
c        decide if the boxes intersect
c
        eps=xmax1-xmin1
        if(eps .gt. xmax2-xmin2) eps=xmax2-xmin2
        if(eps .gt. ymax2-ymin2) eps=ymax2-ymin2
        if(eps .gt. zmax2-zmin2) eps=zmax2-zmin2
c
        if(eps .gt. ymax1-ymin1) eps=ymax1-ymin1
        if(eps .gt. zmax1-zmin1) eps=zmax1-zmin1
c
        eps=eps/10000
c
        ifinter=1
        if(xmin1 .gt. xmax2+eps) ifinter=0
        if(xmin2 .gt. xmax1+eps) ifinter=0
c
        if(ymin1 .gt. ymax2+eps) ifinter=0
        if(ymin2 .gt. ymax1+eps) ifinter=0
c
        if(zmin1 .gt. zmax2+eps) ifinter=0
        if(zmin2 .gt. zmax1+eps) ifinter=0
        return
        end
c
c
c
c
c
        subroutine d3tifint2(box1,box2,ifinter)
        implicit real *8 (a-h,o-z)
        integer box1(20),box2(20)
c
c        this subroutine determines if two boxes in the square 
c        intersect or touch.
c
c                input parameters:
c
c  box1 - the integer array describing the first box
c  box1 - the integer array describing the second box
c
c       integer arrays are dimensioned (20), as produced by d3tallb
c
c                output parametes:
c 
c  ifinter - the indicator.
c      ifinter=1 means that the boxes intersect
c      ifinter=0 means that the boxes do not intersect
c
c
        ifinter=1
c
        do i=1,3
        level1=box1(1)
        level2=box2(1)
        ip1=box1(i+1)-1
        ip2=box2(i+1)-1
	if( (ip1+1)*2**(level2-level1) .lt. (ip2  ) ) ifinter=0
	if( (ip1  )*2**(level2-level1) .gt. (ip2+1) ) ifinter=0
        if( ifinter .eq. 0 ) return
        enddo
c
        return
        end
c
c
c
c
c
        subroutine getCenter(centers,corners,ibox,center,corner)
        implicit real *8 (a-h,o-z)
        real *8 centers(3,*),corners(3,8,*),center(3),corner(3,8)
ccc        save
c
        center(1)=centers(1,ibox)        
        center(2)=centers(2,ibox)        
        center(3)=centers(3,ibox)        
c
        do 1200 i=1,8
        corner(1,i)=corners(1,i,ibox)        
        corner(2,i)=corners(2,i,ibox)        
        corner(3,i)=corners(3,i,ibox)        
c
 1200 continue
c
        return
        end

c
c
c
c
c
        subroutine growTree(ier,z,n,ncrit,boxes,maxboxes,
     1    nboxes,iz,laddr,nlev,center0,size,iwork,
     1    ifempty,minlevel,maxlevel)
        implicit real *8 (a-h,o-z)
        integer boxes(20,*),iz(*),laddr(2,*),iwork(*),
     1      is(8),ns(8),
     1      iisons(8),jjsons(8),kksons(8)
        real *8 z(3,*),center0(3),center(3)
        data kksons/1,1,1,1,2,2,2,2/,jjsons/1,1,2,2,1,1,2,2/,
     1      iisons/1,2,1,2,1,2,1,2/
ccc        save
c
c        this subroutine constructs a oct-tree corresponding
c        to the user-specified collection of points in the space
c
c              input parameters:
c
c  z - the set of points in the space
c  n - the number of elements in z
c

c  ncrit - the maximum number of points permitted in a box on 
c        the finest level. in other words, a box will be further
c        subdivided if it contains more than ncrit points.
c  maxboxes - the maximum total number of boxes the subroutine 
c        is permitted to create. if the points z are such that 
c        more boxes are needed, the error return code ier is
c        set to 4, and the execution of the subroutine is
c        terminated.
c  
c              output parameters:
c
c  ier - the error return code.
c    ier=0 means successful execution
c    ier=4 means that the subroutine attempted to create more 
c        than maxboxes boxes
c    ier=16 means that the subroutine attempted to construct more 
c        than 197 levels of subdivision.
c  boxes - an integer array dimensioned (20,nboxes). each 20-element
c        column describes one box, as follows:
c
c       1. level - the level of subdivision on which this box 
c             was constructed; 
c       2, 3, 4  - the coordinates of this box among  all
c             boxes on this level
c       5 - the daddy of this box, identified by it address
c             in array boxes
c       6,7,8,9,10,11,12,13 - the  list of children of this box 
c             (eight of them, and the child is identified by its address
c             in the array boxes; if a box has only one child, only the
c             first of the four child entries is non-zero, etc.)
c       14 - the location in the array iz of the particles 
c             living in this box
c       15 - the number of particles living in this box
c       16 - the location in the array iztarg of the targets
c             living in this box
c       17 - the number of targets living in this box
c       18 - source box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       19 - target box type: 0 - empty, 1 - leaf node, 2 - sub-divided
c       20 - reserved for future use
c
c    important warning: the array boxes has to be dimensioned 
c                       at least (20,maxboxes)!! otherwise, 
c                       the subroutine is likely to bomb, since
c                       it assumes that it has that much space!!!!
c  nboxes - the total number of boxes created
c  iz - the integer array addressing the particles in 
c         all boxes. 
c       explanation: for a box ibox, the particles living in
c         it are:
c
c         (z(1,j),z(2,j),z(3,j)),(z(1,j+1),z(2,j+1),z(3,j+1)),
c         (z(1,j+2),z(2,j+2),z(3,j+3)), . . . 
c         (z(1,j+nj-1),z(2,j+nj-1),z(3,j+nj-1)),
c         (z(1,j+nj),z(2,j+nj),z(3,j+nj)),
c
c         with j=boxes(14,ibox), and nj=boxes(15,ibox)
c        
c  laddr - an integer array dimensioned (2,numlev), containing
c         the map of array boxes, as follows:
c       laddr(1,i) is the location in array boxes of the information
c         pertaining to level=i-1
c       laddr(2,i) is the number of boxes created on the level i-1
c
c  nlev - the maximum level number on which any boxes have 
c         been created. the maximim number possible is 200. 
c         it is recommended that the array laddr above be 
c         dimensioned at least (2,200), in case the user underestimates
c         the number of levels required.
c  center0 - the center of the box on the level 0, containing
c         the whole simulation
c  size - the side of the box on the level 0
c
c                  work arrays:
c
c  iwork - must be at least n+2 integer*4 elements long.
c
c        . . . find the main box containing the whole picture
c
c
        ier=0
        xmin=z(1,1)
        xmax=z(1,1)
        ymin=z(2,1)
        ymax=z(2,1)
        zmin=z(3,1)
        zmax=z(3,1)
c
c        xmin=1.0d50
c        xmax=-xmin
c        ymin=1.0d50
c        ymax=-ymin
c        zmin=1.0d50
c        zmax=-zmin
c
        do 1100 i=1,n
        if(z(1,i) .lt. xmin) xmin=z(1,i)
        if(z(1,i) .gt. xmax) xmax=z(1,i)
        if(z(2,i) .lt. ymin) ymin=z(2,i)
        if(z(2,i) .gt. ymax) ymax=z(2,i)
        if(z(3,i) .lt. zmin) zmin=z(3,i)
        if(z(3,i) .gt. zmax) zmax=z(3,i)
 1100 continue
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
        laddr(1,1)=1
        laddr(2,1)=1
c
        do 1200 i=1,n 
        iz(i)=i
 1200 continue
c
c       recursively (one level after another) subdivide all 
c       boxes till none are left with more than ncrit particles
        maxson=maxboxes
        maxlev=198
        if( maxlevel .lt. maxlev ) maxlev=maxlevel
        ison=1
        nlev=0
        do 3000 level=0,maxlev-1
        laddr(1,level+2)=laddr(1,level+1)+laddr(2,level+1)
        nlevson=0
        idad0=laddr(1,level+1)
        idad1=idad0+laddr(2,level+1)-1
        do 2000 idad=idad0,idad1
c       subdivide the box number idad (if needed)
        numpdad=boxes(15,idad)
        numtdad=boxes(17,idad)
c       ... refine on both sources and targets
        if(numpdad .le. ncrit .and. numtdad .le. ncrit .and.
     $     level .ge. minlevel ) goto 2000
c       ... not a leaf node on sources or targets
        if( numpdad .gt. ncrit .or. numtdad .gt. ncrit ) then
        if( boxes(18,idad) .eq. 1 ) boxes(18,idad)=2
        if( boxes(19,idad) .eq. 1 ) boxes(19,idad)=2
        endif
        ii=boxes(2,idad)
        jj=boxes(3,idad)
        kk=boxes(4,idad)
        call d3tcentf(center0,size,level,ii,jj,kk,center)
        iiz=boxes(14,idad)
        nz=boxes(15,idad)
        call d3tsepa1(center,z,iz(iiz),nz,iwork,is,ns)
c
c       store in array boxes the sons obtained by the routine 
c       d3tsepa1
c
         idadson=6
        do 1600 i=1,8
        if(ns(i) .eq. 0 .and. 
     $     ifempty .ne. 1) goto 1600
        nlevson=nlevson+1
        ison=ison+1
        nlev=level+1
c
c       . . . if the user-provided array boxes is too
c             short - bomb out
c
        if(ison .le. maxson) goto 1400
        ier=4
        return
 1400 continue
c
c        store in array boxes all information about this son
c
        do 1500 lll=6,13
        boxes(lll,ison)=0
 1500 continue
c
        boxes(1,ison)=level+1
        iison=(ii-1)*2+iisons(i)
        jjson=(jj-1)*2+jjsons(i)
        kkson=(kk-1)*2+kksons(i)
        boxes(2,ison)=iison
        boxes(3,ison)=jjson
        boxes(4,ison)=kkson
        boxes(5,ison)=idad        
c
        boxes(14,ison)=is(i)+iiz-1
        boxes(15,ison)=ns(i)
c
        boxes(16,ison)=0
        boxes(17,ison)=0
        if( ns(i) .le. 0 ) boxes(18,ison)=0
        if( ns(i) .gt. 0 ) boxes(18,ison)=1
        boxes(19,ison)=0
        boxes(20,ison)=0
        boxes(idadson,idad)=ison
        idadson=idadson+1
        nboxes=ison
 1600 continue
 2000 continue
        laddr(2,level+2)=nlevson
         if(nlevson .eq. 0) goto 4000
         level1=level
 3000 continue
        if( level1 .ge. 197 ) ier=16
 4000 continue
        nboxes=ison
        return
        end
c
c
c
c
c
        subroutine setCenter(center0,size,boxes,nboxes,
     1      centers,corners)
        implicit real *8 (a-h,o-z)
        integer boxes(20,*)
        real *8 centers(3,*),corners(3,8,*),center(3),center0(3)
ccc        save
c
c       this subroutine produces arrays of centers and 
c       corners for all boxes in the oct-tree structure.
c
c              input parameters:
c
c  center0 - the center of the box on the level 0, containing
c         the whole simulation
c  size - the side of the box on the level 0
c  boxes - an integer array dimensioned (10,nboxes), as produced 
c        by the subroutine d3tallb (see)
c        column describes one box, as follows:
c  nboxes - the total number of boxes created
c
c  
c              output parameters:
c
c  centers - the centers of all boxes in the array boxes
c  corners - the corners of all boxes in the array boxes
c
c       . . . construct the corners for all boxes
c
        x00=center0(1)-size/2
        y00=center0(2)-size/2
        z00=center0(3)-size/2
        do 1400 i=1,nboxes
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
c
        corners(1,5,i)=corners(1,1,i)+side
        corners(1,6,i)=corners(1,5,i)
        corners(1,7,i)=corners(1,5,i)
        corners(1,8,i)=corners(1,5,i)
c
c
        corners(2,1,i)=center(2)-side/2
        corners(2,2,i)=corners(2,1,i)
        corners(2,5,i)=corners(2,1,i)
        corners(2,6,i)=corners(2,1,i)
c
        corners(2,3,i)=corners(2,1,i)+side
        corners(2,4,i)=corners(2,3,i)
        corners(2,7,i)=corners(2,3,i)
        corners(2,8,i)=corners(2,3,i)
c
c
        corners(3,1,i)=center(3)-side/2
        corners(3,3,i)=corners(3,1,i)
        corners(3,5,i)=corners(3,1,i)
        corners(3,7,i)=corners(3,1,i)
c
        corners(3,2,i)=corners(3,1,i)+side
        corners(3,4,i)=corners(3,2,i)
        corners(3,6,i)=corners(3,2,i)
        corners(3,8,i)=corners(3,2,i)
c
 1400 continue
         return
         end
c
c
c
c
c
        subroutine d3tcentf(center0,size,level,i,j,k,center) 
        implicit real *8 (a-h,o-z)
        real *8 center(3),center0(3)
        data level0/-1/
ccc        save
c
c       this subroutine finds the center of the box 
c       number (i,j) on the level level. note that the 
c       box on level 0 is assumed to have the center 
c       center0, and the side size
c
ccc        if(level .eq. level0) goto 1200
        side=size/2**level
        side2=side/2
        x0=center0(1)-size/2
        y0=center0(2)-size/2
        z0=center0(3)-size/2
        level0=level
 1200 continue
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
        subroutine d3tsepa1(cent,z,iz,n,iwork,
     1    is,ns)
        implicit real *8 (a-h,o-z)
        real *8 cent(3),z(3,*)
        integer iz(*),iwork(*),is(*),ns(*)
ccc        save
c
c        this subroutine reorders the particles in a box,
c        so that each of the children occupies a contigious 
c        chunk of array iz
c
c        note that we are using a standard numbering convention 
c        for the children:
c
c
c             5,6     7,8
c    
c                                     <- looking down the z-axis
c             1,2     3,4
c
c
cccc        in the original code, we were using a strange numbering convention 
cccc        for the children:
cccc
cccc             3,4     7,8
cccc    
cccc                                     <- looking down the z-axis
cccc             1,2     5,6
c
c 
c                        input parameters:
c
c  cent - the center of the box to be subdivided
c  z - the list of all points in the box to be subdivided
c  iz - the integer array specifying the transposition already
c       applied to the points z, before the subdivision of 
c       the box into children
c  n - the total number of points in array z
c  
c                        output parameters:
c
c  iz - the integer array specifying the transposition already
c       applied to the points z, after the subdivision of 
c       the box into children
c  is - an integer array of length 8 containing the locations 
c       of the sons in array iz
c  ns - an integer array of length 8 containig the numbers of 
c       elements in the sons
c
c                        work arrays:
c
c  iwork - must be n integer elements long
c
c        . . . separate all particles in this box in x
c
        n1=0
        n2=0
        n3=0
        n4=0
        n5=0
        n6=0
        n7=0
        n8=0
c
        n12=0
        n34=0
        n56=0
        n78=0
c
        n1234=0
        n5678=0
c
ccc        itype=1
ccc        thresh=cent(1)
        itype=3
        thresh=cent(3)
        call d3tsepa0(z,iz,n,itype,thresh,iwork,n1234)
        n5678=n-n1234
c
c       at this point, the contents of sons number 1,2,3,4 are in
c       the part of array iz with numbers 1,2,...n1234
c       the contents of sons number 5,6,7,8  are in
c       the part of array iz with numbers n1234+1,n12+2,...,n
c
c        . . . separate the boxes 1, 2, 3, 4 and boxes 5, 6, 7, 8
c
        itype=2
        thresh=cent(2)
        if(n1234 .ne. 0) 
     1    call d3tsepa0(z,iz,n1234,itype,thresh,iwork,n12)
        n34=n1234-n12
c
        if(n5678 .ne. 0) 
     1    call d3tsepa0(z,iz(n1234+1),n5678,itype,thresh,iwork,n56)
        n78=n5678-n56
c
c       perform the final separation of pairs of sonnies into
c       individual ones
c
ccc        itype=3
ccc     thresh=cent(3)
        itype=1
        thresh=cent(1)
        if(n12 .ne. 0) 
     1    call d3tsepa0(z,iz,n12,itype,thresh,iwork,n1)
        n2=n12-n1
c
        if(n34 .ne. 0) 
     1    call d3tsepa0(z,iz(n12+1),n34,itype,thresh,iwork,n3)
        n4=n34-n3
c
        if(n56 .ne. 0) 
     1    call d3tsepa0(z,iz(n1234+1),n56,itype,thresh,iwork,n5)
        n6=n56-n5
c
        if(n78 .ne. 0) 
     1    call d3tsepa0(z,iz(n1234+n56+1),n78,itype,thresh,iwork,n7)
        n8=n78-n7
c
c
c       store the information about the sonnies in appropriate arrays
c
        is(1)=1
        ns(1)=n1
c
        is(2)=is(1)+ns(1)
        ns(2)=n2
c
        is(3)=is(2)+ns(2)
        ns(3)=n3
c
        is(4)=is(3)+ns(3)
        ns(4)=n4
c
        is(5)=is(4)+ns(4)
        ns(5)=n5
c
        is(6)=is(5)+ns(5)
        ns(6)=n6
c
        is(7)=is(6)+ns(6)
        ns(7)=n7
c
        is(8)=is(7)+ns(7)
        ns(8)=n8
c
        return
        end
c
c
c
c
c
        subroutine d3tsepa0(z,iz,n,itype,thresh,iwork,n1)
        implicit real *8 (a-h,o-z)
        real *8 z(3,*)
        integer iz(*),iwork(*)
ccc        save
c
c       subdivide the points in this box, horizontally
c       or vertically, depending on the parameter itype
c
cccc        call prin2('in d3tsepa0,thresh=*',thresh,1)
c
        i1=0
        i2=0
        do 1400 i=1,n
        j=iz(i)
        if(z(itype,j) .gt. thresh) goto 1200
        i1=i1+1
        iz(i1)=j
        goto 1400
c
 1200 continue
        i2=i2+1
        iwork(i2)=j
 1400 continue
c
        do 1600 i=1,i2
        iz(i1+i)=iwork(i)
 1600 continue
        n1=i1
        return
        end
c
c
c
c
c
        subroutine d3tlinkinit(ier,nboxes,ntypes,lw)
        use arrays, only : listOffset
        data ilists/0/,numele/0/
        ier=0
        ilists=nboxes*ntypes
        do k=1,ntypes
           do i=1,nboxes
              listOffset(i,k)=-1
           enddo
        enddo
        return
c
        entry d3tlinkstor(ier,itype,ibox,nboxes,list,nlist)
c
c       this entry stores dynamically a list of positive numbers 
c       in the storage array w.
c
c                      input parameters:
c
c  itype - the type of the elements being stored
c  ibox - the box to which these elements corresponds
c  list - the list of positive integer elements to be stored
c  nlist - the number of elements in the array list
c  w - the storage area used by this subroutine; must be first 
c          formatted by the entry d3tlinkinit of this subroutine
c          (see above)
c
c                      output parameters:
c
c  ier - error return code; 
c         ier=0 means successful execution
c  w - the storage area used by this subroutine
c
c       . . . if this storage request exceeds the available memory - bomb
c
        ier=0
c
c       store the user-specified list in array w
c
        call d3tlinksto0(itype,ibox,list,nlist,
     $      nboxes,numele)
c
c       augment the amount of storage used 
c        
        return
c
c
c
c
        entry d3tlinkretr(ier,itype,ibox,nboxes,list,nlist)
c
c       this entry retrieves from the storage area  w 
c       a list of positive numbers that has been stored there
c       by the entry d3tlinkstor (see above).
c
c                      input parameters:
c
c  itype - the type of the elements to be retrieved
c  ibox - the box to which these elements correspond
c  w - the storage area from which the information is to be 
c          retrieved
c
c                      output parameters:
c
c  ier - error return code;
c          ier=0 means successful execution, nothing to report
c          ier=4 means that no elements are present of the type
c                  itype and the box ibox
c  list - the list of positive integer elements retrieved 
c  nlist - the number of elements in the array list
c
        call d3tlinkret0(ier,itype,ibox,
     $       list,nboxes,nlist)
c
        return
        end
c
        subroutine d3tlinksto0(itype,ibox,list,nlist,
     $     nboxes,numele)
        use arrays, only : listOffset,lists
        integer list(*)
ccc        save
c
c       this entry stores dynamically a list of positive numbers 
c       in the storage array lists, while entering the information
c       about this event in the array listOffset.
c
c                      input parameters:
c
c  itype - the type of the elements being stored
c  ibox - the box to which these elements corresponds
c  list - the list of positive integer elements to be stored
c  nlist - the number of elements in the array list
c  listOffset - the addressing array for the main storage array lists;
c             it is assumed that it has been formatted by a call 
c             to the entry d3tlinkini0 of this subroutine (see below).
c  nboxes - the total number of boxes indexing the elements 
c           in array lists
c  lists - the main storage area used by this subroutine
c  numele - the number of elements stored in array lists on entry 
c             to this subroutine
c
c                      output parameters:
c
c  listOffset - the addressing array for the main storage array lists;
c             it is assumed that it has been formatted by a call 
c             to the entry d3tlinkini0 of this subroutine (see below).
c  lists - the main storage area used by this subroutine
c  numele - the number of elements stored in array lists on exit
c             from this subroutine
c .........................................................................
c
c        interpretation of the entries in arrays lists, listOffset:
c
c  lists(1,i) - the location in array lists of the preceding
c        element in the list with (box,type) combination as the 
c        user-supplied (ibox,itype)
c     lists(1,i) .leq. 0 means that this is the first element 
c        of its type,
c  lists(2,i) - the box number on the interaction list.
c
c
c  listOffset(ibox,itype) - the address in the array lists of the last element
c        of this type for this box;
c     listOffset(ibox,itype) .leq. 0 means that there are no elements 
c        in the list of this type for this box.
c
c       . . . store the user-supplied list elements in the array lists,
c             and enter information about this change in array listOffset
c
        ilast=listOffset(ibox,itype)
        do 1200 i=1,nlist
        numele=numele+1
        lists(1,numele)=ilast
        lists(2,numele)=list(i)
        ilast=numele
 1200 continue
        listOffset(ibox,itype)=ilast
        return
c
        entry d3tlinkret0(ier,itype,ibox,list,
     $       nboxes,nlist)
        ier=0
        ilast=listOffset(ibox,itype)
        if(ilast.le.0)then
           nlist=0
           ier=4
           return
        endif
        nlist=0
        do i=1,1 000 000 000
           if(lists(2,ilast).gt.0)then       
              nlist=nlist+1
              list(nlist)=lists(2,ilast)
           endif
           ilast=lists(1,ilast)
           if(ilast.le.0) exit
        enddo

        if(nlist.lt.0)then
           ier=4
           return
        endif
        if(nlist.eq.1) return
        do i=1,nlist/2
           j=list(i)
           list(i)=list(nlist-i+1)
           list(nlist-i+1)=j
        enddo
        return
        end
