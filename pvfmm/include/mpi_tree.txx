/**
 * \file mpi_tree.txx
 * \author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * \date 12-11-2010
 * \brief This file contains the implementation of the class MPI_Tree.
 */

#include <omp.h>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <algorithm>
#include <stdint.h>
#include <set>

#include <dtypes.h>
#include <ompUtils.h>
#include <parUtils.h>
#include <mem_mgr.hpp>
#include <mpi_node.hpp>
#include <profile.hpp>

namespace pvfmm{

/**
 * @author Dhairya Malhotra, dhairya.malhotra@gmail.com
 * @date 08 Feb 2011
 */
inline int p2oLocal(Vector<MortonId> & nodes, Vector<MortonId>& leaves,
    unsigned int maxNumPts, unsigned int maxDepth, bool complete) {
  assert(maxDepth<=MAX_DEPTH);

  std::vector<MortonId> leaves_lst;
  unsigned int init_size=leaves.Dim();
  unsigned int num_pts=nodes.Dim();

  MortonId curr_node=leaves[0];
  MortonId last_node=leaves[init_size-1].NextId();
  MortonId next_node;

  unsigned int curr_pt=0;
  unsigned int next_pt=curr_pt+maxNumPts;

  while(next_pt <= num_pts){
    next_node = curr_node.NextId();
    while( next_pt < num_pts && next_node > nodes[next_pt] && curr_node.GetDepth() < maxDepth-1 ){
      curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
      next_node = curr_node.NextId();
    }
    leaves_lst.push_back(curr_node);
    curr_node = next_node;

    unsigned int inc=maxNumPts;
    while(next_pt < num_pts && curr_node > nodes[next_pt]){
      // We have more than maxNumPts points per octant because the node can
      // not be refined any further.
      inc=inc<<1;
      next_pt+=inc;
      if(next_pt > num_pts){
        next_pt = num_pts;
        break;
      }
    }

    curr_pt = std::lower_bound(&nodes[0]+curr_pt,&nodes[0]+next_pt,curr_node,std::less<MortonId>())-&nodes[0];
    if(curr_pt >= num_pts) break;
    next_pt = curr_pt + maxNumPts;
    if(next_pt > num_pts) next_pt = num_pts;
  }
#ifndef NDEBUG
  for(size_t i=0;i<leaves_lst.size();i++){
    size_t a=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i],std::less<MortonId>())-&nodes[0];
    size_t b=std::lower_bound(&nodes[0],&nodes[0]+nodes.Dim(),leaves_lst[i].NextId(),std::less<MortonId>())-&nodes[0];
    assert(b-a<=maxNumPts || leaves_lst[i].GetDepth()==maxDepth-1);
    if(i==leaves_lst.size()-1) assert(b==nodes.Dim() && a<nodes.Dim());
    if(i==0) assert(a==0);
  }
#endif
  if(complete)
  while(curr_node<last_node){
    while( curr_node.NextId() > last_node && curr_node.GetDepth() < maxDepth-1 )
      curr_node = curr_node.getDFD(curr_node.GetDepth()+1);
    leaves_lst.push_back(curr_node);
    curr_node = curr_node.NextId();
  }

  leaves=leaves_lst;
  return 0;
}

inline int points2Octree(const Vector<MortonId>& pt_mid, Vector<MortonId>& nodes,
          unsigned int maxDepth, unsigned int maxNumPts) {

  int myrank=0, np=1;

  // Sort morton id of points.
  Profile::Tic("SortMortonId", true, 10);
  Vector<MortonId> pt_sorted;
  //par::partitionW<MortonId>(pt_mid, NULL, comm);
  par::HyperQuickSort(pt_mid, pt_sorted);
  size_t pt_cnt=pt_sorted.Dim();
  Profile::Toc();

  // Add last few points from next process, to get the boundary octant right.
  Profile::Tic("Comm", true, 10);
  Profile::Toc();

  // Construct local octree.
  Profile::Tic("p2o_local", false, 10);
  Vector<MortonId> nodes_local(1); nodes_local[0]=MortonId();
  p2oLocal(pt_sorted, nodes_local, maxNumPts, maxDepth, myrank==np-1);
  Profile::Toc();

  // Remove duplicate nodes on adjacent processors.
  Profile::Tic("RemoveDuplicates", true, 10);
  {
    size_t node_cnt=nodes_local.Dim();
    MortonId first_node;
    MortonId  last_node=nodes_local[node_cnt-1];
    size_t i=0;
    std::vector<MortonId> node_lst;
    if(myrank){
      while(i<node_cnt && nodes_local[i].getDFD(maxDepth)<first_node) i++; assert(i);
      last_node=nodes_local[i>0?i-1:0].NextId(); // Next MortonId in the tree after first_node.

      while(first_node<last_node){ // Complete nodes between first_node and last_node.
        while(first_node.isAncestor(last_node))
          first_node=first_node.getDFD(first_node.GetDepth()+1);
        if(first_node==last_node) break;
        node_lst.push_back(first_node);
        first_node=first_node.NextId();
      }
    }
    for(;i<node_cnt-(myrank==np-1?0:1);i++) node_lst.push_back(nodes_local[i]);
    nodes=node_lst;
  }
  Profile::Toc();

  // Repartition nodes.
  Profile::Tic("partitionW", false, 10);
  Profile::Toc();

  return 0;
}

template <class TreeNode>
void MPI_Tree<TreeNode>::Initialize(typename Node_t::NodeData* init_data){
  //Initialize root node.
  Profile::Tic("InitRoot",false,5);
  Tree<TreeNode>::Initialize(init_data);
  TreeNode* rnode=this->RootNode();
  assert(this->dim==COORD_DIM);
  Profile::Toc();

  Profile::Tic("Points2Octee",true,5);
  Vector<MortonId> lin_oct;
  { //Get the linear tree.
    // Compute MortonId from pt_coord.
    Vector<MortonId> pt_mid;
    Vector<Real_t>& pt_coord=rnode->pt_coord;
    size_t pt_cnt=pt_coord.Dim()/this->dim;
    pt_mid.Resize(pt_cnt);
    #pragma omp parallel for
    for(size_t i=0;i<pt_cnt;i++){
      pt_mid[i]=MortonId(pt_coord[i*COORD_DIM+0],pt_coord[i*COORD_DIM+1],pt_coord[i*COORD_DIM+2],this->max_depth);
    }
    points2Octree(pt_mid,lin_oct,this->max_depth,init_data->max_pts);
  }
  Profile::Toc();

  Profile::Tic("ScatterPoints",true,5);
  { // Sort and partition point coordinates and values.
    std::vector<Vector<Real_t>*> coord_lst;
    std::vector<Vector<Real_t>*> value_lst;
    std::vector<Vector<size_t>*> scatter_lst;
    rnode->NodeDataVec(coord_lst, value_lst, scatter_lst);
    assert(coord_lst.size()==value_lst.size());
    assert(coord_lst.size()==scatter_lst.size());

    Vector<MortonId> pt_mid;
    Vector<size_t> scatter_index;
    for(size_t i=0;i<coord_lst.size();i++){
      if(!coord_lst[i]) continue;
      Vector<Real_t>& pt_coord=*coord_lst[i];
      { // Compute MortonId from pt_coord.
        size_t pt_cnt=pt_coord.Dim()/this->dim;
        pt_mid.Resize(pt_cnt);
        #pragma omp parallel for
        for(size_t i=0;i<pt_cnt;i++){
          pt_mid[i]=MortonId(pt_coord[i*COORD_DIM+0],pt_coord[i*COORD_DIM+1],pt_coord[i*COORD_DIM+2],this->max_depth);
        }
      }
      par::SortScatterIndex(pt_mid  , scatter_index, comm, &lin_oct[0]);
      par::ScatterForward  (pt_coord, scatter_index);
      if(value_lst[i]!=NULL){
        Vector<Real_t>& pt_value=*value_lst[i];
        par::ScatterForward(pt_value, scatter_index);
      }
      if(scatter_lst[i]!=NULL){
        Vector<size_t>& pt_scatter=*scatter_lst[i];
        pt_scatter=scatter_index;
      }
    }
  }
  Profile::Toc();

  //Initialize the pointer based tree from the linear tree.
  Profile::Tic("PointerTree",false,5);
  { // Construct the pointer tree from lin_oct
    int omp_p=omp_get_max_threads();

    // Partition nodes between threads
    rnode->SetGhost(false);
    for(int i=0;i<omp_p;i++){
      size_t idx=(lin_oct.Dim()*i)/omp_p;
      Node_t* n=FindNode(lin_oct[idx], true);
      assert(n->GetMortonId()==lin_oct[idx]);
      UNUSED(n);
    }

    #pragma omp parallel for
    for(int i=0;i<omp_p;i++){
      size_t a=(lin_oct.Dim()* i   )/omp_p;
      size_t b=(lin_oct.Dim()*(i+1))/omp_p;

      size_t idx=a;
      Node_t* n=FindNode(lin_oct[idx], false);
      if(a==0) n=rnode;
      while(n!=NULL && (idx<b || i==omp_p-1)){
        n->SetGhost(false);
        MortonId dn=n->GetMortonId();
        if(idx<b && dn.isAncestor(lin_oct[idx])){
          if(n->IsLeaf()) n->Subdivide();
        }else if(idx<b && dn==lin_oct[idx]){
          if(!n->IsLeaf()) n->Truncate();
          assert(n->IsLeaf());
          idx++;
        }else{
          n->Truncate();
          n->SetGhost(true);
        }
        n=this->PreorderNxt(n);
      }
      //assert(idx==b); // TODO: Check why this fails
    }
  }
  Profile::Toc();
}


template <class TreeNode>
TreeNode* MPI_Tree<TreeNode>::FindNode(MortonId& key, bool subdiv,  TreeNode* start){
  int num_child=1UL<<this->Dim();
  Node_t* n=start;
  if(n==NULL) n=this->RootNode();
  while(n->GetMortonId()<key && (!n->IsLeaf()||subdiv)){
    if(n->IsLeaf() && !n->IsGhost()) n->Subdivide();
    if(n->IsLeaf()) break;
    for(int j=0;j<num_child;j++){
      if(((Node_t*)n->Child(j))->GetMortonId().NextId()>key){
        n=(Node_t*)n->Child(j);
        break;
      }
    }
  }
  assert(!subdiv || n->IsGhost() || n->GetMortonId()==key);
  return n;
}


//list must be sorted.
inline int lineariseList(std::vector<MortonId> & list, MPI_Comm comm) {
  int rank,size;
  MPI_Comm_rank(comm,&rank);
  MPI_Comm_size(comm,&size);

  //Remove empty processors...
  int new_rank, new_size;
  MPI_Comm   new_comm;
  MPI_Comm_split(comm, (list.empty()?0:1), rank, &new_comm);

  MPI_Comm_rank (new_comm, &new_rank);
  MPI_Comm_size (new_comm, &new_size);
  if(!list.empty()) {
    //Send the last octant to the next processor.
    MortonId lastOctant = list[list.size()-1];
    MortonId lastOnPrev;

    MPI_Request recvRequest;
    MPI_Request sendRequest;

    if(new_rank > 0) {
      MPI_Irecv(&lastOnPrev, 1, par::Mpi_datatype<MortonId>::value(), new_rank-1, 1, new_comm, &recvRequest);
    }
    if(new_rank < (new_size-1)) {
      MPI_Issend( &lastOctant, 1, par::Mpi_datatype<MortonId>::value(), new_rank+1, 1, new_comm,  &sendRequest);
    }

    if(new_rank > 0) {
      std::vector<MortonId> tmp(list.size()+1);
      for(size_t i = 0; i < list.size(); i++) {
        tmp[i+1] = list[i];
      }

      MPI_Status statusWait;
      MPI_Wait(&recvRequest, &statusWait);
      tmp[0] = lastOnPrev;

      list.swap(tmp);
    }

    {// Remove duplicates and ancestors.
      std::vector<MortonId> tmp;
      if(!(list.empty())) {
        for(unsigned int i = 0; i < (list.size()-1); i++) {
          if( (!(list[i].isAncestor(list[i+1]))) && (list[i] != list[i+1]) ) {
            tmp.push_back(list[i]);
          }
        }
        if(new_rank == (new_size-1)) {
          tmp.push_back(list[list.size()-1]);
        }
      }
      list.swap(tmp);
    }

    if(new_rank < (new_size-1)) {
      MPI_Status statusWait;
      MPI_Wait(&sendRequest, &statusWait);
    }
  }//not empty procs only

  // Free new_comm
  MPI_Comm_free(&new_comm);

  return 1;
}//end fn.


template <class TreeNode>
void MPI_Tree<TreeNode>::SetColleagues(BoundaryType bndry, Node_t* node){
  int n1=(int)pvfmm::pow<unsigned int>(3,this->Dim());
  int n2=(int)pvfmm::pow<unsigned int>(2,this->Dim());

  if(node==NULL){
    Node_t* curr_node=this->PreorderFirst();
    if(curr_node!=NULL){
      if(bndry==Periodic){
        for(int i=0;i<n1;i++)
          curr_node->SetColleague(curr_node,i);
      }else{
        curr_node->SetColleague(curr_node,(n1-1)/2);
      }
      curr_node=this->PreorderNxt(curr_node);
    }

    Vector<std::vector<Node_t*> > nodes(MAX_DEPTH);
    while(curr_node!=NULL){
      nodes[curr_node->Depth()].push_back(curr_node);
      curr_node=this->PreorderNxt(curr_node);
    }
    for(size_t i=0;i<MAX_DEPTH;i++){
      size_t j0=nodes[i].size();
      Node_t** nodes_=&nodes[i][0];
      #pragma omp parallel for
      for(size_t j=0;j<j0;j++){
        SetColleagues(bndry, nodes_[j]);
      }
    }

  }else{
    /* //This is slower
    Node_t* root_node=this->RootNode();
    int d=node->Depth();
    Real_t* c0=node->Coord();
    Real_t s=pvfmm::pow<Real_t>(0.5,d);
    Real_t c[COORD_DIM];
    int idx=0;
    for(int i=-1;i<=1;i++)
    for(int j=-1;j<=1;j++)
    for(int k=-1;k<=1;k++){
      c[0]=c0[0]+s*0.5+s*k;
      c[1]=c0[1]+s*0.5+s*j;
      c[2]=c0[2]+s*0.5+s*i;
      if(bndry==Periodic){
        if(c[0]<0.0) c[0]+=1.0;
        if(c[0]>1.0) c[0]-=1.0;
        if(c[1]<1.0) c[1]+=1.0;
        if(c[1]>1.0) c[1]-=1.0;
        if(c[2]<1.0) c[2]+=1.0;
        if(c[2]>1.0) c[2]-=1.0;
      }
      node->SetColleague(NULL,idx);
      if(c[0]<1.0 && c[0]>0.0)
      if(c[1]<1.0 && c[1]>0.0)
      if(c[2]<1.0 && c[2]>0.0){
        MortonId m(c,d);
        Node_t* nbr=FindNode(m,false,root_node);
        while(nbr->Depth()>d) nbr=(Node_t*)nbr->Parent();
        if(nbr->Depth()==d) node->SetColleague(nbr,idx);
      }
      idx++;
    }
    /*/
    Node_t* parent_node;
    Node_t* tmp_node1;
    Node_t* tmp_node2;

    for(int i=0;i<n1;i++)node->SetColleague(NULL,i);
    parent_node=(Node_t*)node->Parent();
    if(parent_node==NULL) return;

    int l=node->Path2Node();
    for(int i=0;i<n1;i++){ //For each coll of the parent
      tmp_node1=(Node_t*)parent_node->Colleague(i);
      if(tmp_node1!=NULL)
      if(!tmp_node1->IsLeaf()){
        for(int j=0;j<n2;j++){ //For each child
          tmp_node2=(Node_t*)tmp_node1->Child(j);
          if(tmp_node2!=NULL){

            bool flag=true;
            int a=1,b=1,new_indx=0;
            for(int k=0;k<this->Dim();k++){
              int indx_diff=(((i/b)%3)-1)*2+((j/a)%2)-((l/a)%2);
              if(-1>indx_diff || indx_diff>1) flag=false;
              new_indx+=(indx_diff+1)*b;
              a*=2;b*=3;
            }
            if(flag){
              node->SetColleague(tmp_node2,new_indx);
            }
          }
        }
      }
    }// */
  }

}

template <class TreeNode>
void IsShared(std::vector<TreeNode*>& nodes, MortonId* m1, MortonId* m2, BoundaryType bndry, std::vector<char>& shared_flag){
  MortonId mm1, mm2;
  if(m1!=NULL) mm1=m1->getDFD();
  if(m2!=NULL) mm2=m2->getDFD();
  shared_flag.resize(nodes.size());
  int omp_p=omp_get_max_threads();

  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    size_t a=((j  )*nodes.size())/omp_p;
    size_t b=((j+1)*nodes.size())/omp_p;
    std::vector<MortonId> nbr_lst;
    for(size_t i=a;i<b;i++){
      shared_flag[i]=false;
      TreeNode* node=nodes[i];
      assert(node!=NULL);
      if(node->Depth()<2){
        shared_flag[i]=true;
        continue;
      }
      node->GetMortonId().NbrList(nbr_lst, node->Depth()-1, bndry==Periodic);
      for(size_t k=0;k<nbr_lst.size();k++){
        MortonId n1=nbr_lst[k]         .getDFD();
        MortonId n2=nbr_lst[k].NextId().getDFD();
        if(m1==NULL || n2>mm1)
          if(m2==NULL || n1<mm2){
            shared_flag[i]=true;
            break;
          }
      }
    }
  }
}

inline void IsShared(std::vector<PackedData>& nodes, MortonId* m1, MortonId* m2, BoundaryType bndry, std::vector<char>& shared_flag){
  MortonId mm1, mm2;
  if(m1!=NULL) mm1=m1->getDFD();
  if(m2!=NULL) mm2=m2->getDFD();
  shared_flag.resize(nodes.size());
  int omp_p=omp_get_max_threads();

  #pragma omp parallel for
  for(int j=0;j<omp_p;j++){
    size_t a=((j  )*nodes.size())/omp_p;
    size_t b=((j+1)*nodes.size())/omp_p;
    std::vector<MortonId> nbr_lst;
    for(size_t i=a;i<b;i++){
      shared_flag[i]=false;
      MortonId* node=(MortonId*)nodes[i].data;
      assert(node!=NULL);
      if(node->GetDepth()<2){
        shared_flag[i]=true;
        continue;
      }
      node->NbrList(nbr_lst, node->GetDepth()-1, bndry==Periodic);
      for(size_t k=0;k<nbr_lst.size();k++){
        MortonId n1=nbr_lst[k]         .getDFD();
        MortonId n2=nbr_lst[k].NextId().getDFD();
        if(m1==NULL || n2>mm1)
          if(m2==NULL || n1<mm2){
            shared_flag[i]=true;
            break;
          }
      }
    }
  }
}


inline bool isLittleEndian(){
  uint16_t number = 0x1;
  uint8_t *numPtr = (uint8_t*)&number;
  return (numPtr[0] == 1);
}

template <class TreeNode>
void MPI_Tree<TreeNode>::Write2File(const char* fname, int lod){
  typedef double VTKReal_t;

  int myrank, np;
  MPI_Comm_size(*Comm(),&np);
  MPI_Comm_rank(*Comm(),&myrank);

  VTUData_t<VTKReal_t> vtu_data;
  TreeNode::VTU_Data(vtu_data, this->GetNodeList(), lod);

  std::vector<VTKReal_t>&               coord=vtu_data.coord;
  std::vector<std::string>&             name =vtu_data.name;
  std::vector<std::vector<VTKReal_t> >& value=vtu_data.value;

  std::vector<int32_t>& connect=vtu_data.connect;
  std::vector<int32_t>& offset =vtu_data.offset;
  std::vector<uint8_t>& types  =vtu_data.types;

  int pt_cnt=coord.size()/COORD_DIM;
  int cell_cnt=types.size();

  std::vector<int32_t> mpi_rank;  //MPI_Rank at points.
  int new_myrank=myrank;//rand();
  mpi_rank.resize(pt_cnt,new_myrank);

  //Open file for writing.
  std::stringstream vtufname;
  vtufname<<fname<<std::setfill('0')<<std::setw(6)<<myrank<<".vtu";
  std::ofstream vtufile;
  vtufile.open(vtufname.str().c_str());
  if(vtufile.fail()) return;

  //Proceed to write to file.
  size_t data_size=0;
  vtufile<<"<?xml version=\"1.0\"?>\n";
  if(isLittleEndian()) vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">\n";
  else                 vtufile<<"<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n";
  //===========================================================================
  vtufile<<"  <UnstructuredGrid>\n";
  vtufile<<"    <Piece NumberOfPoints=\""<<pt_cnt<<"\" NumberOfCells=\""<<cell_cnt<<"\">\n";

  //---------------------------------------------------------------------------
  vtufile<<"      <Points>\n";
  vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal_t)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+coord.size()*sizeof(VTKReal_t);
  vtufile<<"      </Points>\n";
  //---------------------------------------------------------------------------
  vtufile<<"      <PointData>\n";
  for(size_t i=0;i<name.size();i++) if(value[i].size()){ // value
    vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal_t)*8<<"\" NumberOfComponents=\""<<value[i].size()/pt_cnt<<"\" Name=\""<<name[i]<<"\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+value[i].size()*sizeof(VTKReal_t);
  }
  { // mpi_rank
    vtufile<<"        <DataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
    data_size+=sizeof(uint32_t)+mpi_rank.size()*sizeof(int32_t);
  }
  vtufile<<"      </PointData>\n";
  //---------------------------------------------------------------------------
  //---------------------------------------------------------------------------
  vtufile<<"      <Cells>\n";
  vtufile<<"        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+connect.size()*sizeof(int32_t);
  vtufile<<"        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+offset.size() *sizeof(int32_t);
  vtufile<<"        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  data_size+=sizeof(uint32_t)+types.size()  *sizeof(uint8_t);
  vtufile<<"      </Cells>\n";
  //---------------------------------------------------------------------------
  //vtufile<<"      <CellData>\n";
  //vtufile<<"        <DataArray type=\"Float"<<sizeof(VTKReal_t)*8<<"\" Name=\"Velocity\" format=\"appended\" offset=\""<<data_size<<"\" />\n";
  //vtufile<<"      </CellData>\n";
  //---------------------------------------------------------------------------

  vtufile<<"    </Piece>\n";
  vtufile<<"  </UnstructuredGrid>\n";
  //===========================================================================
  vtufile<<"  <AppendedData encoding=\"raw\">\n";
  vtufile<<"    _";

  int32_t block_size;
  block_size=coord   .size()*sizeof(VTKReal_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&coord   [0], coord   .size()*sizeof(VTKReal_t));
  for(size_t i=0;i<name.size();i++) if(value[i].size()){ // value
    block_size=value[i].size()*sizeof(VTKReal_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&value[i][0], value[i].size()*sizeof(VTKReal_t));
  }
  block_size=mpi_rank.size()*sizeof(  int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&mpi_rank[0], mpi_rank.size()*sizeof(  int32_t));

  block_size=connect.size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&connect[0], connect.size()*sizeof(int32_t));
  block_size=offset .size()*sizeof(int32_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&offset [0], offset .size()*sizeof(int32_t));
  block_size=types  .size()*sizeof(uint8_t); vtufile.write((char*)&block_size, sizeof(int32_t)); vtufile.write((char*)&types  [0], types  .size()*sizeof(uint8_t));

  vtufile<<"\n";
  vtufile<<"  </AppendedData>\n";
  //===========================================================================
  vtufile<<"</VTKFile>\n";
  vtufile.close();


  if(myrank) return;
  std::stringstream pvtufname;
  pvtufname<<fname<<".pvtu";
  std::ofstream pvtufile;
  pvtufile.open(pvtufname.str().c_str());
  if(pvtufile.fail()) return;
  pvtufile<<"<?xml version=\"1.0\"?>\n";
  pvtufile<<"<VTKFile type=\"PUnstructuredGrid\">\n";
  pvtufile<<"  <PUnstructuredGrid GhostLevel=\"0\">\n";
  pvtufile<<"      <PPoints>\n";
  pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal_t)*8<<"\" NumberOfComponents=\""<<COORD_DIM<<"\" Name=\"Position\"/>\n";
  pvtufile<<"      </PPoints>\n";
  pvtufile<<"      <PPointData>\n";
  for(size_t i=0;i<name.size();i++) if(value[i].size()){ // value
    pvtufile<<"        <PDataArray type=\"Float"<<sizeof(VTKReal_t)*8<<"\" NumberOfComponents=\""<<value[i].size()/pt_cnt<<"\" Name=\""<<name[i]<<"\"/>\n";
  }
  { // mpi_rank
    pvtufile<<"        <PDataArray type=\"Int32\" NumberOfComponents=\"1\" Name=\"mpi_rank\"/>\n";
  }
  pvtufile<<"      </PPointData>\n";
  {
    // Extract filename from path.
    std::stringstream vtupath;
    vtupath<<'/'<<fname;
    std::string pathname = vtupath.str();
    unsigned found = pathname.find_last_of("/\\");
    std::string fname_ = pathname.substr(found+1);
    //char *fname_ = (char*)strrchr(vtupath.str().c_str(), '/') + 1;
    //std::string fname_ = boost::filesystem::path(fname).filename().string().
    for(int i=0;i<np;i++) pvtufile<<"      <Piece Source=\""<<fname_<<std::setfill('0')<<std::setw(6)<<i<<".vtu\"/>\n";
  }
  pvtufile<<"  </PUnstructuredGrid>\n";
  pvtufile<<"</VTKFile>\n";
  pvtufile.close();
}


template <class TreeNode>
const std::vector<MortonId>& MPI_Tree<TreeNode>::GetMins(){
  Node_t* n=this->PreorderFirst();
  while(n!=NULL){
    if(!n->IsGhost() && n->IsLeaf()) break;
    n=this->PreorderNxt(n);
  }

  MortonId my_min;
  my_min=n->GetMortonId();

  int np;
  MPI_Comm_size(*Comm(),&np);
  mins.resize(np);

  MPI_Allgather(&my_min , 1, par::Mpi_datatype<MortonId>::value(),
                &mins[0], 1, par::Mpi_datatype<MortonId>::value(), *Comm());

  return mins;
}

}//end namespace
