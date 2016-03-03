#include <vector>

#ifndef _PVFMM_MORTONID_HPP_
#define _PVFMM_MORTONID_HPP_

namespace pvfmm{

#ifndef MAX_DEPTH
#define MAX_DEPTH 30
#endif

#if MAX_DEPTH < 7
#define UINT_T uint8_t
#define  INT_T  int8_t
#elif MAX_DEPTH < 15
#define UINT_T uint16_t
#define  INT_T  int16_t
#elif MAX_DEPTH < 31
#define UINT_T uint32_t
#define  INT_T  int32_t
#elif MAX_DEPTH < 63
#define UINT_T uint64_t
#define  INT_T  int64_t
#endif

class MortonId{

 public:

  inline MortonId():x(0), y(0), z(0), depth(0){}

  inline MortonId(MortonId m, uint8_t depth_):x(m.x), y(m.y), z(m.z), depth(depth_){
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  template <class T>
  inline MortonId(T x_f,T y_f, T z_f, uint8_t depth_=MAX_DEPTH) : depth(depth_) {
    static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
    x=(UINT_T)floor(x_f*max_int);
    y=(UINT_T)floor(y_f*max_int);
    z=(UINT_T)floor(z_f*max_int);
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  template <class T>
  inline MortonId(T* coord, uint8_t depth_=MAX_DEPTH) : depth(depth_){
    static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
    x=(UINT_T)floor(coord[0]*max_int);
    y=(UINT_T)floor(coord[1]*max_int);
    z=(UINT_T)floor(coord[2]*max_int);
    assert(depth<=MAX_DEPTH);
    UINT_T mask=~((((UINT_T)1)<<(MAX_DEPTH-depth))-1);
    x=x & mask;
    y=y & mask;
    z=z & mask;
  }

  inline unsigned int GetDepth() const {
    return depth;
  }

  template <class T>
  inline void GetCoord(T* coord) {
    static UINT_T max_int=((UINT_T)1)<<(MAX_DEPTH);
    static T s=1.0/((T)max_int);
    coord[0]=x*s;
    coord[1]=y*s;
    coord[2]=z*s;
  }

  inline MortonId NextId() const {
    MortonId m=*this;
    UINT_T mask=((UINT_T)1)<<(MAX_DEPTH-depth);
    int i;
    for(i=depth;i>=0;i--){
      m.x=(m.x^mask);
      if((m.x & mask))
	break;
      m.y=(m.y^mask);
      if((m.y & mask))
	break;
      m.z=(m.z^mask);
      if((m.z & mask))
	break;
      mask=(mask<<1);
    }
    if(i<0) i=0;
    m.depth=(uint8_t)i;
    return m;
  }

  inline MortonId getAncestor(uint8_t ancestor_level) const {
    MortonId m=*this;
    m.depth=ancestor_level;
    UINT_T mask=(((UINT_T)1)<<(MAX_DEPTH))-(((UINT_T)1)<<(MAX_DEPTH-ancestor_level));
    m.x=(m.x & mask);
    m.y=(m.y & mask);
    m.z=(m.z & mask);
    return m;
  }

  inline MortonId getDFD(uint8_t level=MAX_DEPTH) const {
    MortonId m=*this;
    m.depth=level;
    return m;
  }

  void NbrList(std::vector<MortonId>& nbrs,uint8_t level, int periodic) const{
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

  std::vector<MortonId> Children() const {
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

  inline int operator<(const MortonId& m) const {
    if(x==m.x && y==m.y && z==m.z) return depth<m.depth;
    UINT_T x_=(x^m.x);
    UINT_T y_=(y^m.y);
    UINT_T z_=(z^m.z);
    if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
      return z<m.z;
    if(y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_))
      return y<m.y;
    return x<m.x;
  }

  inline int operator>(const MortonId& m) const {
    if(x==m.x && y==m.y && z==m.z) return depth>m.depth;
    UINT_T x_=(x^m.x);
    UINT_T y_=(y^m.y);
    UINT_T z_=(z^m.z);
    if((z_>x_ || ((z_^x_)<x_ && (z_^x_)<z_)) && (z_>y_ || ((z_^y_)<y_ && (z_^y_)<z_)))
      return z>m.z;
    if((y_>x_ || ((y_^x_)<x_ && (y_^x_)<y_)))
      return y>m.y;
    return x>m.x;
  }

  inline int operator==(const MortonId& m) const {
    return (x==m.x && y==m.y && z==m.z && depth==m.depth);
  }

  inline int isAncestor(MortonId const & other) const {
    return other.depth>depth && other.getAncestor(depth)==*this;
  }

  friend std::ostream& operator<<(std::ostream& out, const MortonId & mid);

 private:

  UINT_T x,y,z;
  uint8_t depth;

};

}//end namespace

#endif //_PVFMM_MORTONID_HPP_
