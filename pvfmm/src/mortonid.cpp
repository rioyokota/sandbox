#include <iostream>
#include <cassert>
#include <cstdlib>
#include <cmath>
#include <mortonid.hpp>

namespace pvfmm{

void MortonId::NbrList(std::vector<MortonId>& nbrs, uint8_t level, int periodic) const{
  nbrs.clear();
  static unsigned int dim=3;
  static unsigned int nbr_cnt=powf(3,dim);
  static UINT_T maxCoord=(((UINT_T)1)<<(MAX_DEPTH));

  UINT_T mask=maxCoord-(((UINT_T)1)<<(MAX_DEPTH-level));
  UINT_T pX=x & mask;
  UINT_T pY=y & mask;
  UINT_T pZ=z & mask;

  MortonId mid_tmp;
  mask=(((UINT_T)1)<<(MAX_DEPTH-level));
  for(int i=0; i<nbr_cnt; i++){
    INT_T dX = ((i/1)%3-1)*mask;
    INT_T dY = ((i/3)%3-1)*mask;
    INT_T dZ = ((i/9)%3-1)*mask;
    INT_T newX=(INT_T)pX+dX;
    INT_T newY=(INT_T)pY+dY;
    INT_T newZ=(INT_T)pZ+dZ;
    if(!periodic){
      if(newX>=0 && newX<(INT_T)maxCoord)
      if(newY>=0 && newY<(INT_T)maxCoord)
      if(newZ>=0 && newZ<(INT_T)maxCoord){
        mid_tmp.x=newX; mid_tmp.y=newY; mid_tmp.z=newZ;
        mid_tmp.depth=level;
        nbrs.push_back(mid_tmp);
      }
    }else{
      if(newX<0) newX+=maxCoord; if(newX>=(INT_T)maxCoord) newX-=maxCoord;
      if(newY<0) newY+=maxCoord; if(newY>=(INT_T)maxCoord) newY-=maxCoord;
      if(newZ<0) newZ+=maxCoord; if(newZ>=(INT_T)maxCoord) newZ-=maxCoord;
      mid_tmp.x=newX; mid_tmp.y=newY; mid_tmp.z=newZ;
      mid_tmp.depth=level;
      nbrs.push_back(mid_tmp);
    }
  }
}

std::vector<MortonId> MortonId::Children() const{
  static int dim=3;
  static int c_cnt=(1UL<<dim);
  static UINT_T maxCoord=(((UINT_T)1)<<(MAX_DEPTH));
  std::vector<MortonId> child(c_cnt);

  UINT_T mask=maxCoord-(((UINT_T)1)<<(MAX_DEPTH-depth));
  UINT_T pX=x & mask;
  UINT_T pY=y & mask;
  UINT_T pZ=z & mask;

  mask=(((UINT_T)1)<<(MAX_DEPTH-(depth+1)));
  for(int i=0; i<c_cnt; i++){
    child[i].x=pX+mask*((i/1)%2);
    child[i].y=pY+mask*((i/2)%2);
    child[i].z=pZ+mask*((i/4)%2);
    child[i].depth=(uint8_t)(depth+1);
  }
  return child;
}

}//end namespace
