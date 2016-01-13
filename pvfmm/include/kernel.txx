#include <cmath>
#include <cstdlib>
#include <vector>

#include <mem_mgr.hpp>
#include <profile.hpp>
#include <vector.hpp>
#include <matrix.hpp>
#include <precomp_mat.hpp>
#include <intrin_wrapper.hpp>
#include <cheb_utils.hpp>

namespace pvfmm{

/**
 * \brief Constructor.
 */
template <class T>
Kernel<T>::Kernel(Ker_t poten, Ker_t dbl_poten, const char* name, int dim_, std::pair<int,int> k_dim,
                  size_t dev_poten, size_t dev_dbl_poten){
  dim=dim_;
  ker_dim[0]=k_dim.first;
  ker_dim[1]=k_dim.second;
  ker_poten=poten;
  dbl_layer_poten=dbl_poten;
  ker_name=std::string(name);

  dev_ker_poten=dev_poten;
  dev_dbl_layer_poten=dev_dbl_poten;

  k_s2m=NULL;
  k_s2l=NULL;
  k_s2t=NULL;
  k_m2m=NULL;
  k_m2l=NULL;
  k_m2t=NULL;
  k_l2l=NULL;
  k_l2t=NULL;
  vol_poten=NULL;

  scale_invar=true;
  src_scal.Resize(ker_dim[0]); src_scal.SetZero();
  trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
  perm_vec.Resize(Perm_Count);
  for(size_t p_type=0;p_type<C_Perm;p_type++){
    perm_vec[p_type       ]=Permutation<T>(ker_dim[0]);
    perm_vec[p_type+C_Perm]=Permutation<T>(ker_dim[1]);
  }
  init=false;
}

/**
 * \brief Initialize the kernel.
 */
template <class T>
void Kernel<T>::Initialize(bool verbose) const{
  if(init) return;
  init=true;

  T eps=1.0;
  while(eps+(T)1.0>1.0) eps*=0.5;

  T scal=1.0;
  if(ker_dim[0]*ker_dim[1]>0){ // Determine scaling
    Matrix<T> M_scal(ker_dim[0],ker_dim[1]);
    size_t N=1024;
    T eps_=N*eps;

    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*COORD_DIM);
    Matrix<T> M1(N,ker_dim[0]*ker_dim[1]);
    while(true){
      T abs_sum=0;
      for(size_t i=0;i<N/2;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=pvfmm::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*COORD_DIM+0]=x*scal;
        trg_coord1[i*COORD_DIM+1]=y*scal;
        trg_coord1[i*COORD_DIM+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=pvfmm::sqrt<T>(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*COORD_DIM+0]=x*1.0/scal;
        trg_coord1[i*COORD_DIM+1]=y*1.0/scal;
        trg_coord1[i*COORD_DIM+2]=z*1.0/scal;
      }
      for(size_t i=0;i<N;i++){
        BuildMatrix(&src_coord [          0], 1,
                    &trg_coord1[i*COORD_DIM], 1, &(M1[i][0]));
        for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
          abs_sum+=pvfmm::fabs<T>(M1[i][j]);
        }
      }
      if(abs_sum>pvfmm::sqrt<T>(eps) || scal<eps) break;
      scal=scal*0.5;
    }

    std::vector<T> trg_coord2(N*COORD_DIM);
    Matrix<T> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N*COORD_DIM;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*COORD_DIM], 1, &(M2[i][0]));
    }

    for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
      T dot11=0, dot12=0, dot22=0;
      for(size_t j=0;j<N;j++){
        dot11+=M1[j][i]*M1[j][i];
        dot12+=M1[j][i]*M2[j][i];
        dot22+=M2[j][i]*M2[j][i];
      }
      T max_val=std::max<T>(dot11,dot22);
      if(dot11>max_val*eps &&
         dot22>max_val*eps ){
        T s=dot12/dot11;
        M_scal[0][i]=pvfmm::log<T>(s)/pvfmm::log<T>(2.0);
        T err=pvfmm::sqrt<T>(0.5*(dot22/dot11)/(s*s)-0.5);
        if(err>eps_){
          scale_invar=false;
          M_scal[0][i]=0.0;
        }
        //assert(M_scal[0][i]>=0.0); // Kernel function must decay
      }else if(dot11>max_val*eps ||
               dot22>max_val*eps ){
        scale_invar=false;
        M_scal[0][i]=0.0;
      }else{
        M_scal[0][i]=-1;
      }
    }

    src_scal.Resize(ker_dim[0]); src_scal.SetZero();
    trg_scal.Resize(ker_dim[1]); trg_scal.SetZero();
    if(scale_invar){
      Matrix<T> b(ker_dim[0]*ker_dim[1]+1,1); b.SetZero();
      mem::memcopy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(T));

      Matrix<T> M(ker_dim[0]*ker_dim[1]+1,ker_dim[0]+ker_dim[1]); M.SetZero();
      M[ker_dim[0]*ker_dim[1]][0]=1;
      for(size_t i0=0;i0<ker_dim[0];i0++)
      for(size_t i1=0;i1<ker_dim[1];i1++){
        size_t j=i0*ker_dim[1]+i1;
        if(fabs(b[j][0])>=0){
          M[j][ 0+        i0]=1;
          M[j][i1+ker_dim[0]]=1;
        }
      }
      Matrix<T> x=M.pinv()*b;

      for(size_t i=0;i<ker_dim[0];i++){
        src_scal[i]=x[i][0];
      }
      for(size_t i=0;i<ker_dim[1];i++){
        trg_scal[i]=x[ker_dim[0]+i][0];
      }

      for(size_t i0=0;i0<ker_dim[0];i0++)
      for(size_t i1=0;i1<ker_dim[1];i1++){
        if(M_scal[i0][i1]>=0){
          if(pvfmm::fabs<T>(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
            scale_invar=false;
          }
        }
      }
    }

    if(!scale_invar){
      src_scal.SetZero();
      trg_scal.SetZero();
      //std::cout<<ker_name<<" not-scale-invariant\n";
    }
  }
  if(ker_dim[0]*ker_dim[1]>0){ // Determine symmetry
    size_t N=1024;
    T eps_=N*eps;
    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*COORD_DIM);
    std::vector<T> trg_coord2(N*COORD_DIM);
    for(size_t i=0;i<N/2;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=pvfmm::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x*scal;
      trg_coord1[i*COORD_DIM+1]=y*scal;
      trg_coord1[i*COORD_DIM+2]=z*scal;
    }
    for(size_t i=N/2;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=pvfmm::sqrt<T>(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*COORD_DIM+0]=x*1.0/scal;
      trg_coord1[i*COORD_DIM+1]=y*1.0/scal;
      trg_coord1[i*COORD_DIM+2]=z*1.0/scal;
    }

    for(size_t p_type=0;p_type<C_Perm;p_type++){ // For each symmetry transform

      switch(p_type){ // Set trg_coord2
        case ReflecX:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]=-trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case ReflecY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]=-trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case ReflecZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]=-trg_coord1[i*COORD_DIM+2];
          }
          break;
        case SwapXY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
          break;
        case SwapXZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+2];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+0];
          }
          break;
        default:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*COORD_DIM+0]= trg_coord1[i*COORD_DIM+0];
            trg_coord2[i*COORD_DIM+1]= trg_coord1[i*COORD_DIM+1];
            trg_coord2[i*COORD_DIM+2]= trg_coord1[i*COORD_DIM+2];
          }
      }

      Matrix<long long> M11, M22;
      {
        Matrix<T> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
        Matrix<T> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
        for(size_t i=0;i<N;i++){
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord1[i*COORD_DIM], 1, &(M1[i][0]));
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord2[i*COORD_DIM], 1, &(M2[i][0]));
        }

        Matrix<T> dot11(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot11.SetZero();
        Matrix<T> dot12(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot12.SetZero();
        Matrix<T> dot22(ker_dim[0]*ker_dim[1],ker_dim[0]*ker_dim[1]);dot22.SetZero();
        std::vector<T> norm1(ker_dim[0]*ker_dim[1]);
        std::vector<T> norm2(ker_dim[0]*ker_dim[1]);
        {
          for(size_t k=0;k<N;k++)
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
          for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
            dot11[i][j]+=M1[k][i]*M1[k][j];
            dot12[i][j]+=M1[k][i]*M2[k][j];
            dot22[i][j]+=M2[k][i]*M2[k][j];
          }
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
            norm1[i]=pvfmm::sqrt<T>(dot11[i][i]);
            norm2[i]=pvfmm::sqrt<T>(dot22[i][i]);
          }
          for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++)
          for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
            dot11[i][j]/=(norm1[i]*norm1[j]);
            dot12[i][j]/=(norm1[i]*norm2[j]);
            dot22[i][j]/=(norm2[i]*norm2[j]);
          }
        }

        long long flag=1;
        M11.Resize(ker_dim[0],ker_dim[1]); M11.SetZero();
        M22.Resize(ker_dim[0],ker_dim[1]); M22.SetZero();
        for(size_t i=0;i<ker_dim[0]*ker_dim[1];i++){
          if(norm1[i]>eps_ && M11[0][i]==0){
            for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
              if(pvfmm::fabs<T>(norm1[i]-norm1[j])<eps_ && pvfmm::fabs<T>(pvfmm::fabs<T>(dot11[i][j])-1.0)<eps_){
                M11[0][j]=(dot11[i][j]>0?flag:-flag);
              }
              if(pvfmm::fabs<T>(norm1[i]-norm2[j])<eps_ && pvfmm::fabs<T>(pvfmm::fabs<T>(dot12[i][j])-1.0)<eps_){
                M22[0][j]=(dot12[i][j]>0?flag:-flag);
              }
            }
            flag++;
          }
        }
      }

      Matrix<long long> P1, P2;
      { // P1
        Matrix<long long>& P=P1;
        Matrix<long long>  M1=M11;
        Matrix<long long>  M2=M22;
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
          std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
          std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
        }
        P.Resize(M1.Dim(0),M1.Dim(0));
        for(size_t i=0;i<M1.Dim(0);i++)
        for(size_t j=0;j<M1.Dim(0);j++){
          P[i][j]=1;
          for(size_t k=0;k<M1.Dim(1);k++)
          if(M1[i][k]!=M2[j][k]){
            P[i][j]=0;
            break;
          }
        }
      }
      { // P2
        Matrix<long long>& P=P2;
        Matrix<long long>  M1=M11.Transpose();
        Matrix<long long>  M2=M22.Transpose();
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
          std::sort(&M1[i][0],&M1[i][M1.Dim(1)]);
          std::sort(&M2[i][0],&M2[i][M2.Dim(1)]);
        }
        P.Resize(M1.Dim(0),M1.Dim(0));
        for(size_t i=0;i<M1.Dim(0);i++)
        for(size_t j=0;j<M1.Dim(0);j++){
          P[i][j]=1;
          for(size_t k=0;k<M1.Dim(1);k++)
          if(M1[i][k]!=M2[j][k]){
            P[i][j]=0;
            break;
          }
        }
      }

      std::vector<Permutation<long long> > P1vec, P2vec;
      { // P1vec
        Matrix<long long>& Pmat=P1;
        std::vector<Permutation<long long> >& Pvec=P1vec;

        Permutation<long long> P(Pmat.Dim(0));
        Vector<PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PERM_INT_T> perm_tmp;
        while(true){ // Next permutation
          perm_tmp=perm;
          std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.Dim());
          for(size_t i=0;i<perm_tmp.Dim();i++){
            if(perm_tmp[i]!=i) break;
            if(i==perm_tmp.Dim()-1){
              Pvec.push_back(P);
            }
          }

          bool last=false;
          for(size_t i=0;i<P.Dim();i++){
            PERM_INT_T tmp=perm[i];
            for(size_t j=perm[i]+1;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(perm[i]>tmp) break;
            for(size_t j=0;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(i==P.Dim()-1) last=true;
          }
          if(last) break;
        }
      }
      { // P2vec
        Matrix<long long>& Pmat=P2;
        std::vector<Permutation<long long> >& Pvec=P2vec;

        Permutation<long long> P(Pmat.Dim(0));
        Vector<PERM_INT_T>& perm=P.perm;
        perm.SetZero();

        // First permutation
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }

        Vector<PERM_INT_T> perm_tmp;
        while(true){ // Next permutation
          perm_tmp=perm;
          std::sort(&perm_tmp[0],&perm_tmp[0]+perm_tmp.Dim());
          for(size_t i=0;i<perm_tmp.Dim();i++){
            if(perm_tmp[i]!=i) break;
            if(i==perm_tmp.Dim()-1){
              Pvec.push_back(P);
            }
          }

          bool last=false;
          for(size_t i=0;i<P.Dim();i++){
            PERM_INT_T tmp=perm[i];
            for(size_t j=perm[i]+1;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(perm[i]>tmp) break;
            for(size_t j=0;j<P.Dim();j++){
              if(Pmat[i][j]){
                perm[i]=j;
                break;
              }
            }
            if(i==P.Dim()-1) last=true;
          }
          if(last) break;
        }
      }

      { // Find pairs which acutally work (neglect scaling)
        std::vector<Permutation<long long> > P1vec_, P2vec_;
        Matrix<long long>  M1=M11;
        Matrix<long long>  M2=M22;
        for(size_t i=0;i<M1.Dim(0);i++){
          for(size_t j=0;j<M1.Dim(1);j++){
            if(M1[i][j]<0) M1[i][j]=-M1[i][j];
            if(M2[i][j]<0) M2[i][j]=-M2[i][j];
          }
        }

        Matrix<long long> M;
        for(size_t i=0;i<P1vec.size();i++)
        for(size_t j=0;j<P2vec.size();j++){
          M=P1vec[i]*M2*P2vec[j];
          for(size_t k=0;k<M.Dim(0)*M.Dim(1);k++){
            if(M[0][k]!=M1[0][k]) break;
            if(k==M.Dim(0)*M.Dim(1)-1){
              P1vec_.push_back(P1vec[i]);
              P2vec_.push_back(P2vec[j]);
            }
          }
        }

        P1vec=P1vec_;
        P2vec=P2vec_;
      }

      Permutation<T> P1_, P2_;
      { // Find pairs which acutally work
        for(size_t k=0;k<P1vec.size();k++){
          Permutation<long long> P1=P1vec[k];
          Permutation<long long> P2=P2vec[k];
          Matrix<long long>  M1=   M11   ;
          Matrix<long long>  M2=P1*M22*P2;

          Matrix<T> M(M1.Dim(0)*M1.Dim(1)+1,M1.Dim(0)+M1.Dim(1));
          M.SetZero(); M[M1.Dim(0)*M1.Dim(1)][0]=1.0;
          for(size_t i=0;i<M1.Dim(0);i++)
          for(size_t j=0;j<M1.Dim(1);j++){
            size_t k=i*M1.Dim(1)+j;
            M[k][          i]= M1[i][j];
            M[k][M1.Dim(0)+j]=-M2[i][j];
          }
          M=M.pinv();
          { // Construct new permutation
            Permutation<long long> P1_(M1.Dim(0));
            Permutation<long long> P2_(M1.Dim(1));
            for(size_t i=0;i<M1.Dim(0);i++){
              P1_.scal[i]=(M[i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
            }
            for(size_t i=0;i<M1.Dim(1);i++){
              P2_.scal[i]=(M[M1.Dim(0)+i][M1.Dim(0)*M1.Dim(1)]>0?1:-1);
            }
            P1=P1_*P1 ;
            P2=P2 *P2_;
          }

          bool done=true;
          Matrix<long long> Merr=P1*M22*P2-M11;
          for(size_t i=0;i<Merr.Dim(0)*Merr.Dim(1);i++){
            if(Merr[0][i]){
              done=false;
              break;
            }
          }
          if(done){
            P1_=Permutation<T>(P1.Dim());
            P2_=Permutation<T>(P2.Dim());
            for(size_t i=0;i<P1.Dim();i++){
              P1_.perm[i]=P1.perm[i];
              P1_.scal[i]=P1.scal[i];
            }
            for(size_t i=0;i<P2.Dim();i++){
              P2_.perm[i]=P2.perm[i];
              P2_.scal[i]=P2.scal[i];
            }
            break;
          }
        }
      }

      //std::cout<<P1_<<'\n';
      //std::cout<<P2_<<'\n';
      perm_vec[p_type       ]=P1_.Transpose();
      perm_vec[p_type+C_Perm]=P2_;
    }

    for(size_t i=0;i<2*C_Perm;i++){
      if(perm_vec[i].Dim()==0){
        perm_vec.Resize(0);
        std::cout<<"no-symmetry for: "<<ker_name<<'\n';
        break;
      }
    }
  }

  if(verbose){ // Display kernel information
    std::cout<<"\n";
    std::cout<<"Kernel Name    : "<<ker_name<<'\n';
    std::cout<<"Precision      : "<<(double)eps<<'\n';
    std::cout<<"Symmetry       : "<<(perm_vec.Dim()>0?"yes":"no")<<'\n';
    std::cout<<"Scale Invariant: "<<(scale_invar?"yes":"no")<<'\n';
    if(scale_invar && ker_dim[0]*ker_dim[1]>0){
      std::cout<<"Scaling Matrix :\n";
      Matrix<T> Src(ker_dim[0],1);
      Matrix<T> Trg(1,ker_dim[1]);
      for(size_t i=0;i<ker_dim[0];i++) Src[i][0]=pvfmm::pow<T>(2.0,src_scal[i]);
      for(size_t i=0;i<ker_dim[1];i++) Trg[0][i]=pvfmm::pow<T>(2.0,trg_scal[i]);
      std::cout<<Src*Trg;
    }
    if(ker_dim[0]*ker_dim[1]>0){ // Accuracy of multipole expansion
      std::cout<<"Multipole Error: ";
      for(T rad=1.0; rad>1.0e-2; rad*=0.5){
        int m=8; // multipole order

        std::vector<T> equiv_surf;
        std::vector<T> check_surf;
        for(int i0=0;i0<m;i0++){
          for(int i1=0;i1<m;i1++){
            for(int i2=0;i2<m;i2++){
              if(i0==  0 || i1==  0 || i2==  0 ||
                 i0==m-1 || i1==m-1 || i2==m-1){

                // Range: [-1/3,1/3]^3
                T x=((T)2*i0-(m-1))/(m-1)/3;
                T y=((T)2*i1-(m-1))/(m-1)/3;
                T z=((T)2*i2-(m-1))/(m-1)/3;

                equiv_surf.push_back(x*RAD0*rad);
                equiv_surf.push_back(y*RAD0*rad);
                equiv_surf.push_back(z*RAD0*rad);

                check_surf.push_back(x*RAD1*rad);
                check_surf.push_back(y*RAD1*rad);
                check_surf.push_back(z*RAD1*rad);
              }
            }
          }
        }
        size_t n_equiv=equiv_surf.size()/COORD_DIM;
        size_t n_check=equiv_surf.size()/COORD_DIM;

        size_t n_src=m*m;
        size_t n_trg=m*m;
        std::vector<T> src_coord;
        std::vector<T> trg_coord;
        for(size_t i=0;i<n_src*COORD_DIM;i++){
          src_coord.push_back((2*drand48()-1)/3*rad);
        }
        for(size_t i=0;i<n_trg;i++){
          T x,y,z,r;
          do{
            x=(drand48()-0.5);
            y=(drand48()-0.5);
            z=(drand48()-0.5);
            r=pvfmm::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          trg_coord.push_back(x/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(y/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          trg_coord.push_back(z/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
        }

        Matrix<T> M_s2c(n_src*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                    &check_surf[0], n_check, &(M_s2c[0][0]));

        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));
        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                     &trg_coord[0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_s2t(n_src*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                     &trg_coord[0], n_trg  , &(M_s2t[0][0]));

        Matrix<T> M=(M_s2c*M_c2e0)*(M_c2e1*M_e2t)-M_s2t;
        T max_error=0, max_value=0;
        for(size_t i=0;i<M.Dim(0);i++)
        for(size_t j=0;j<M.Dim(1);j++){
          max_error=std::max<T>(max_error,pvfmm::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,pvfmm::fabs<T>(M_s2t[i][j]));
        }

        std::cout<<(double)(max_error/max_value)<<' ';
        if(scale_invar) break;
      }
      std::cout<<"\n";
    }
    if(ker_dim[0]*ker_dim[1]>0){ // Accuracy of local expansion
      std::cout<<"Local-exp Error: ";
      for(T rad=1.0; rad>1.0e-2; rad*=0.5){
        int m=8; // multipole order

        std::vector<T> equiv_surf;
        std::vector<T> check_surf;
        for(int i0=0;i0<m;i0++){
          for(int i1=0;i1<m;i1++){
            for(int i2=0;i2<m;i2++){
              if(i0==  0 || i1==  0 || i2==  0 ||
                 i0==m-1 || i1==m-1 || i2==m-1){

                // Range: [-1/3,1/3]^3
                T x=((T)2*i0-(m-1))/(m-1)/3;
                T y=((T)2*i1-(m-1))/(m-1)/3;
                T z=((T)2*i2-(m-1))/(m-1)/3;

                equiv_surf.push_back(x*RAD1*rad);
                equiv_surf.push_back(y*RAD1*rad);
                equiv_surf.push_back(z*RAD1*rad);

                check_surf.push_back(x*RAD0*rad);
                check_surf.push_back(y*RAD0*rad);
                check_surf.push_back(z*RAD0*rad);
              }
            }
          }
        }
        size_t n_equiv=equiv_surf.size()/COORD_DIM;
        size_t n_check=equiv_surf.size()/COORD_DIM;

        size_t n_src=m*m;
        size_t n_trg=m*m;
        std::vector<T> src_coord;
        std::vector<T> trg_coord;
        for(size_t i=0;i<n_trg*COORD_DIM;i++){
          trg_coord.push_back((2*drand48()-1)/3*rad);
        }
        for(size_t i=0;i<n_src;i++){
          T x,y,z,r;
          do{
            x=(drand48()-0.5);
            y=(drand48()-0.5);
            z=(drand48()-0.5);
            r=pvfmm::sqrt<T>(x*x+y*y+z*z);
          }while(r==0.0);
          src_coord.push_back(x/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(y/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
          src_coord.push_back(z/r*pvfmm::sqrt<T>((T)COORD_DIM)*rad*(1.0+drand48()));
        }

        Matrix<T> M_s2c(n_src*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                    &check_surf[0], n_check, &(M_s2c[0][0]));

        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));
        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                     &trg_coord[0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_s2t(n_src*ker_dim[0],n_trg*ker_dim[1]);
        BuildMatrix( &src_coord[0], n_src,
                     &trg_coord[0], n_trg  , &(M_s2t[0][0]));

        Matrix<T> M=(M_s2c*M_c2e0)*(M_c2e1*M_e2t)-M_s2t;
        T max_error=0, max_value=0;
        for(size_t i=0;i<M.Dim(0);i++)
        for(size_t j=0;j<M.Dim(1);j++){
          max_error=std::max<T>(max_error,pvfmm::fabs<T>(M    [i][j]));
          max_value=std::max<T>(max_value,pvfmm::fabs<T>(M_s2t[i][j]));
        }

        std::cout<<(double)(max_error/max_value)<<' ';
        if(scale_invar) break;
      }
      std::cout<<"\n";
    }
    if(vol_poten && ker_dim[0]*ker_dim[1]>0){ // Check if the volume potential is consistent with integral of kernel.
      int m=8; // multipole order
      std::vector<T> equiv_surf;
      std::vector<T> check_surf;
      std::vector<T> trg_coord;
      for(size_t i=0;i<m*COORD_DIM;i++){
        trg_coord.push_back(drand48()+1.0);
      }
      for(int i0=0;i0<m;i0++){
        for(int i1=0;i1<m;i1++){
          for(int i2=0;i2<m;i2++){
            if(i0==  0 || i1==  0 || i2==  0 ||
               i0==m-1 || i1==m-1 || i2==m-1){

              // Range: [-1/2,1/2]^3
              T x=((T)2*i0-(m-1))/(m-1)/2;
              T y=((T)2*i1-(m-1))/(m-1)/2;
              T z=((T)2*i2-(m-1))/(m-1)/2;

              equiv_surf.push_back(x*RAD1+1.5);
              equiv_surf.push_back(y*RAD1+1.5);
              equiv_surf.push_back(z*RAD1+1.5);

              check_surf.push_back(x*RAD0+1.5);
              check_surf.push_back(y*RAD0+1.5);
              check_surf.push_back(z*RAD0+1.5);
            }
          }
        }
      }
      size_t n_equiv=equiv_surf.size()/COORD_DIM;
      size_t n_check=equiv_surf.size()/COORD_DIM;
      size_t n_trg  =trg_coord .size()/COORD_DIM;

      Matrix<T> M_local, M_analytic;
      Matrix<T> T_local, T_analytic;
      { // Compute local expansions M_local, T_local
        Matrix<T> M_near(ker_dim[0],n_check*ker_dim[1]);
        Matrix<T> T_near(ker_dim[0],n_trg  *ker_dim[1]);
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0;i<n_check;i++){ // Compute near-interaction for operator M_near
          std::vector<T> M_=cheb_integ<T>(0, &check_surf[i*3], 3.0, *this);
          for(size_t j=0; j<ker_dim[0]; j++)
            for(int k=0; k<ker_dim[1]; k++)
              M_near[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
        }
        #pragma omp parallel for schedule(dynamic)
        for(size_t i=0;i<n_trg;i++){ // Compute near-interaction for targets T_near
          std::vector<T> M_=cheb_integ<T>(0, &trg_coord[i*3], 3.0, *this);
          for(size_t j=0; j<ker_dim[0]; j++)
            for(int k=0; k<ker_dim[1]; k++)
              T_near[j][i*ker_dim[1]+k] = M_[j+k*ker_dim[0]];
        }

        { // M_local = M_analytic - M_near
          M_analytic.ReInit(ker_dim[0],n_check*ker_dim[1]); M_analytic.SetZero();
          vol_poten(&check_surf[0],n_check,&M_analytic[0][0]);
          M_local=M_analytic-M_near;
        }
        { // T_local = T_analytic - T_near
          T_analytic.ReInit(ker_dim[0],n_trg  *ker_dim[1]); T_analytic.SetZero();
          vol_poten(&trg_coord[0],n_trg,&T_analytic[0][0]);
          T_local=T_analytic-T_near;
        }
      }

      Matrix<T> T_err;
      { // Now we should be able to compute T_local from M_local
        Matrix<T> M_e2c(n_equiv*ker_dim[0],n_check*ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &check_surf[0], n_check, &(M_e2c[0][0]));

        Matrix<T> M_e2t(n_equiv*ker_dim[0],n_trg  *ker_dim[1]);
        BuildMatrix(&equiv_surf[0], n_equiv,
                    &trg_coord [0], n_trg  , &(M_e2t[0][0]));

        Matrix<T> M_c2e0, M_c2e1;
        {
          Matrix<T> U,S,V;
          M_e2c.SVD(U,S,V);
          T eps=1, max_S=0;
          while(eps*(T)0.5+(T)1.0>1.0) eps*=0.5;
          for(size_t i=0;i<std::min(S.Dim(0),S.Dim(1));i++){
            if(pvfmm::fabs<T>(S[i][i])>max_S) max_S=pvfmm::fabs<T>(S[i][i]);
          }
          for(size_t i=0;i<S.Dim(0);i++) S[i][i]=(S[i][i]>eps*max_S*4?1.0/S[i][i]:0.0);
          M_c2e0=V.Transpose()*S;
          M_c2e1=U.Transpose();
        }

        T_err=(M_local*M_c2e0)*(M_c2e1*M_e2t)-T_local;
      }
      { // Print relative error
        T err_sum=0, analytic_sum=0;
        for(size_t i=0;i<T_err     .Dim(0)*T_err     .Dim(1);i++)      err_sum+=pvfmm::fabs<T>(T_err     [0][i]);
        for(size_t i=0;i<T_analytic.Dim(0)*T_analytic.Dim(1);i++) analytic_sum+=pvfmm::fabs<T>(T_analytic[0][i]);
        std::cout<<"Volume Error   : "<<err_sum/analytic_sum<<"\n";
      }
    }
    std::cout<<"\n";
  }

  { // Initialize auxiliary FMM kernels
    if(!k_s2m) k_s2m=this;
    if(!k_s2l) k_s2l=this;
    if(!k_s2t) k_s2t=this;
    if(!k_m2m) k_m2m=this;
    if(!k_m2l) k_m2l=this;
    if(!k_m2t) k_m2t=this;
    if(!k_l2l) k_l2l=this;
    if(!k_l2t) k_l2t=this;

    assert(k_s2t->ker_dim[0]==ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2l->ker_dim[0]);
    assert(k_s2m->ker_dim[0]==k_s2t->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2l->ker_dim[0]);
    assert(k_m2m->ker_dim[0]==k_m2t->ker_dim[0]);
    assert(k_l2l->ker_dim[0]==k_l2t->ker_dim[0]);

    assert(k_s2t->ker_dim[1]==ker_dim[1]);
    assert(k_s2m->ker_dim[1]==k_m2m->ker_dim[1]);
    assert(k_s2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_m2l->ker_dim[1]==k_l2l->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_m2t->ker_dim[1]);
    assert(k_s2t->ker_dim[1]==k_l2t->ker_dim[1]);

    k_s2m->Initialize(verbose);
    k_s2l->Initialize(verbose);
    k_s2t->Initialize(verbose);
    k_m2m->Initialize(verbose);
    k_m2l->Initialize(verbose);
    k_m2t->Initialize(verbose);
    k_l2l->Initialize(verbose);
    k_l2t->Initialize(verbose);
  }
}

/**
 * \brief Compute the transformation matrix (on the source strength vector)
 * to get potential at target coordinates due to sources at the given
 * coordinates.
 * \param[in] r_src Coordinates of source points.
 * \param[in] src_cnt Number of source points.
 * \param[in] r_trg Coordinates of target points.
 * \param[in] trg_cnt Number of target points.
 * \param[out] k_out Output array with potential values.
 */
template <class T>
void Kernel<T>::BuildMatrix(T* r_src, int src_cnt,
                 T* r_trg, int trg_cnt, T* k_out) const{
  int dim=3; //Only supporting 3D
  memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(T));
  for(int i=0;i<src_cnt;i++) //TODO Optimize this.
    for(int j=0;j<ker_dim[0];j++){
      std::vector<T> v_src(ker_dim[0],0);
      v_src[j]=1.0;
      ker_poten(&r_src[i*dim], 1, &v_src[0], 1, r_trg, trg_cnt,
                &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]], NULL);
    }
}


/**
 * \brief Generic kernel which rearranges data for vectorization, calls the
 * actual uKernel and copies data to the output array in the original order.
 */
template <class Real_t, int SRC_DIM, int TRG_DIM, void (*uKernel)(Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&)>
void generic_kernel(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg, mem::MemoryManager* mem_mgr){
  assert(dof==1);
  int VecLen=8;
  if(sizeof(Real_t)==sizeof( float)) VecLen=8;
  if(sizeof(Real_t)==sizeof(double)) VecLen=4;

  #define STACK_BUFF_SIZE 4096
  Real_t stack_buff[STACK_BUFF_SIZE+MEM_ALIGN];
  Real_t* buff=NULL;

  Matrix<Real_t> src_coord;
  Matrix<Real_t> src_value;
  Matrix<Real_t> trg_coord;
  Matrix<Real_t> trg_value;
  { // Rearrange data in src_coord, src_coord, trg_coord, trg_value
    size_t src_cnt_, trg_cnt_; // counts after zero padding
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;

    size_t buff_size=src_cnt_*(COORD_DIM+SRC_DIM)+
                     trg_cnt_*(COORD_DIM+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){ // Allocate buff
      buff=mem::aligned_new<Real_t>(buff_size, mem_mgr);
    }

    Real_t* buff_ptr=buff;
    if(!buff_ptr){ // use stack_buff
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(Real_t*)ptr;
    }
    src_coord.ReInit(COORD_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=COORD_DIM*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(COORD_DIM, trg_cnt_,buff_ptr,false);  buff_ptr+=COORD_DIM*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);//buff_ptr+=  TRG_DIM*trg_cnt_;
    { // Set src_coord
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<COORD_DIM;j++){
          src_coord[j][i]=r_src[i*COORD_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<COORD_DIM;j++){
          src_coord[j][i]=0;
        }
      }
    }
    { // Set src_value
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=v_src[i*SRC_DIM+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<SRC_DIM;j++){
          src_value[j][i]=0;
        }
      }
    }
    { // Set trg_coord
      size_t i=0;
      for(   ;i<trg_cnt ;i++){
        for(size_t j=0;j<COORD_DIM;j++){
          trg_coord[j][i]=r_trg[i*COORD_DIM+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<COORD_DIM;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    { // Set trg_value
      size_t i=0;
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<TRG_DIM;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }
  uKernel(src_coord,src_value,trg_coord,trg_value);
  { // Set v_trg
    for(size_t i=0;i<trg_cnt ;i++){
      for(size_t j=0;j<TRG_DIM;j++){
        v_trg[i*TRG_DIM+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){ // Free memory: buff
    mem::aligned_delete<Real_t>(buff);
  }
}


////////////////////////////////////////////////////////////////////////////////
////////                   LAPLACE KERNEL                               ////////
////////////////////////////////////////////////////////////////////////////////

/**
 * \brief Green's function for the Poisson's equation. Kernel tensor
 * dimension = 1x1.
 */
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_poten_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 1000
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = 1.0/(4*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[0][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }

  { // Add FLOPS
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(12+4*(NWTN_ITER)));
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 1, 1, laplace_poten_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}

// Laplace double layer potential.
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_dbl_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sn0=             bcast_intrin<Vec_t>(&src_value[0][s]) ;
        Vec_t sn1=             bcast_intrin<Vec_t>(&src_value[1][s]) ;
        Vec_t sn2=             bcast_intrin<Vec_t>(&src_value[2][s]) ;
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[3][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);

        Vec_t rdotn=            mul_intrin(sn0,dx);
        rdotn=add_intrin(rdotn, mul_intrin(sn1,dy));
        rdotn=add_intrin(rdotn, mul_intrin(sn2,dz));

        sv=mul_intrin(sv,rdotn);
        tv=add_intrin(tv,mul_intrin(r3inv,sv));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }

  { // Add FLOPS
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(20+4*(NWTN_ITER)));
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_dbl_poten(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 4, 1, laplace_dbl_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}


// Laplace grdient kernel.
template <class Real_t, class Vec_t=Real_t, Vec_t (*RSQRT_INTRIN)(Vec_t)=rsqrt_intrin0<Vec_t> >
void laplace_grad_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);

  //// Number of newton iterations
  size_t NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin0<Vec_t,Real_t>) NWTN_ITER=0;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin1<Vec_t,Real_t>) NWTN_ITER=1;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin2<Vec_t,Real_t>) NWTN_ITER=2;
  if(RSQRT_INTRIN==(Vec_t (*)(Vec_t))rsqrt_intrin3<Vec_t,Real_t>) NWTN_ITER=3;

  Real_t nwtn_scal=1; // scaling factor for newton iterations
  for(int i=0;i<NWTN_ITER;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*const_pi<Real_t>());

  size_t src_cnt_=src_coord.Dim(1);
  size_t trg_cnt_=trg_coord.Dim(1);
  for(size_t sblk=0;sblk<src_cnt_;sblk+=SRC_BLK){
    size_t src_cnt=src_cnt_-sblk;
    if(src_cnt>SRC_BLK) src_cnt=SRC_BLK;
    for(size_t t=0;t<trg_cnt_;t+=VecLen){
      Vec_t tx=load_intrin<Vec_t>(&trg_coord[0][t]);
      Vec_t ty=load_intrin<Vec_t>(&trg_coord[1][t]);
      Vec_t tz=load_intrin<Vec_t>(&trg_coord[2][t]);
      Vec_t tv0=zero_intrin<Vec_t>();
      Vec_t tv1=zero_intrin<Vec_t>();
      Vec_t tv2=zero_intrin<Vec_t>();
      for(size_t s=sblk;s<sblk+src_cnt;s++){
        Vec_t dx=sub_intrin(tx,bcast_intrin<Vec_t>(&src_coord[0][s]));
        Vec_t dy=sub_intrin(ty,bcast_intrin<Vec_t>(&src_coord[1][s]));
        Vec_t dz=sub_intrin(tz,bcast_intrin<Vec_t>(&src_coord[2][s]));
        Vec_t sv=              bcast_intrin<Vec_t>(&src_value[0][s]) ;

        Vec_t r2=        mul_intrin(dx,dx) ;
        r2=add_intrin(r2,mul_intrin(dy,dy));
        r2=add_intrin(r2,mul_intrin(dz,dz));

        Vec_t rinv=RSQRT_INTRIN(r2);
        Vec_t r3inv=mul_intrin(mul_intrin(rinv,rinv),rinv);

        sv=mul_intrin(sv,r3inv);
        tv0=add_intrin(tv0,mul_intrin(sv,dx));
        tv1=add_intrin(tv1,mul_intrin(sv,dy));
        tv2=add_intrin(tv2,mul_intrin(sv,dz));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv0=add_intrin(mul_intrin(tv0,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      tv1=add_intrin(mul_intrin(tv1,oofp),load_intrin<Vec_t>(&trg_value[1][t]));
      tv2=add_intrin(mul_intrin(tv2,oofp),load_intrin<Vec_t>(&trg_value[2][t]));
      store_intrin(&trg_value[0][t],tv0);
      store_intrin(&trg_value[1][t],tv1);
      store_intrin(&trg_value[2][t],tv2);
    }
  }

  { // Add FLOPS
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*(19+4*(NWTN_ITER)));
  }
  #undef SRC_BLK
}

template <class T, int newton_iter=0>
void laplace_grad(T* r_src, int src_cnt, T* v_src, int dof, T* r_trg, int trg_cnt, T* v_trg, mem::MemoryManager* mem_mgr){
  #define LAP_KER_NWTN(nwtn) if(newton_iter==nwtn) \
        generic_kernel<Real_t, 1, 3, laplace_grad_uKernel<Real_t,Vec_t, rsqrt_intrin##nwtn<Vec_t,Real_t> > > \
            ((Real_t*)r_src, src_cnt, (Real_t*)v_src, dof, (Real_t*)r_trg, trg_cnt, (Real_t*)v_trg, mem_mgr)
  #define LAPLACE_KERNEL LAP_KER_NWTN(0); LAP_KER_NWTN(1); LAP_KER_NWTN(2); LAP_KER_NWTN(3);

  if(mem::TypeTraits<T>::ID()==mem::TypeTraits<float>::ID()){
    typedef float Real_t;
    #if defined __AVX__
      #define Vec_t __m256
    #elif defined __SSE3__
      #define Vec_t __m128
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else if(mem::TypeTraits<T>::ID()==mem::TypeTraits<double>::ID()){
    typedef double Real_t;
    #if defined __AVX__
      #define Vec_t __m256d
    #elif defined __SSE3__
      #define Vec_t __m128d
    #else
      #define Vec_t Real_t
    #endif
    LAPLACE_KERNEL;
    #undef Vec_t
  }else{
    typedef T Real_t;
    #define Vec_t Real_t
    LAPLACE_KERNEL;
    #undef Vec_t
  }

  #undef LAP_KER_NWTN
  #undef LAPLACE_KERNEL
}

template<class T> const Kernel<T>& LaplaceKernel<T>::gradient(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,1>, laplace_dbl_poten<T,1> >("laplace"     , 3, std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad <T,1>                         >("laplace_grad", 3, std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}

template<> inline const Kernel<double>& LaplaceKernel<double>::gradient(){
  typedef double T;
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten<T,2>, laplace_dbl_poten<T,2> >("laplace"     , 3, std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad <T,2>                         >("laplace_grad", 3, std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}

}//end namespace
