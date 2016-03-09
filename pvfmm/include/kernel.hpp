#ifndef _PVFMM_FMM_KERNEL_HPP_
#define _PVFMM_FMM_KERNEL_HPP_

namespace pvfmm{

template <class T>
struct Kernel{
  public:

  typedef void (*Ker_t)(T* r_src, int src_cnt, T* v_src, int dof,
                        T* r_trg, int trg_cnt, T* k_out);

  typedef void (*VolPoten)(const T* coord, int n, T* out);

  Kernel(Ker_t poten, const char* name, std::pair<int,int> k_dim);

  void Initialize(bool verbose=false) const;

  void BuildMatrix(T* r_src, int src_cnt,
                   T* r_trg, int trg_cnt, T* k_out) const;

  int ker_dim[2];
  std::string ker_name;
  Ker_t ker_poten;

  mutable bool init;
  mutable bool scale_invar;
  mutable Vector<T> src_scal;
  mutable Vector<T> trg_scal;
  mutable Vector<Permutation<T> > perm_vec;

  mutable const Kernel<T>* k_s2m;
  mutable const Kernel<T>* k_s2l;
  mutable const Kernel<T>* k_s2t;
  mutable const Kernel<T>* k_m2m;
  mutable const Kernel<T>* k_m2l;
  mutable const Kernel<T>* k_m2t;
  mutable const Kernel<T>* k_l2l;
  mutable const Kernel<T>* k_l2t;
  mutable VolPoten vol_poten;

};

template <class T>
T machine_eps(){
  T eps=1.0;
  while(eps+(T)1.0>1.0) eps*=0.5;
  return eps;
}

template <class T>
inline void cheb_poly(int d, const T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(fabs(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(fabs(in[i])<=1?1.0:0);
      out[i+n]=(fabs(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(fabs(in[j])<=1?in[j]:0);
      T y0=(fabs(in[j])<=1?1.0:0);
      out[j]=y0;
      out[j+n]=x;

      T y1=x;
      T* y2=&out[2*n+j];
      for(int i=2;i<=d;i++){
        *y2=2*x*y1-y0;
        y0=y1;
        y1=*y2;
        y2=&y2[n];
      }
    }
  }
}

template <class T>
void quad_rule(int n, T* x, T* w){
  static std::vector<Vector<T> > x_lst(10000);
  static std::vector<Vector<T> > w_lst(10000);
  assert(n<10000);
  bool done=false;
  #pragma omp critical (QUAD_RULE)
  if(x_lst[n].Dim()>0){
    Vector<T>& x_=x_lst[n];
    Vector<T>& w_=w_lst[n];
    for(int i=0;i<n;i++){
      x[i]=x_[i];
      w[i]=w_[i];
    }
    done=true;
  }
  if(done) return;
  Vector<T> x_(n);
  Vector<T> w_(n);
  {
    for(int i=0;i<n;i++){
      x_[i]=-cos((T)(2.0*i+1.0)/(2.0*n)*M_PI);
      w_[i]=0;
    }
    Matrix<T> M(n,n);
    cheb_poly(n-1, &x_[0], n, &M[0][0]);
    for(size_t i=0;i<n;i++) M[0][i]/=2.0;

    std::vector<T> w_sample(n,0);
    for(long i=0;i<n;i+=2) w_sample[i]=-((T)2.0/(i+1)/(i-1));
    for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++){
      M[i][j]*=w_sample[i];
    }
    for(size_t i=0;i<n;i++)
    for(size_t j=0;j<n;j++){
      w_[j]+=M[i][j]*2/n;
    }
  }
  #pragma omp critical (QUAD_RULE)
  {
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);
}

template <class T>
std::vector<T> integ_pyramid(int m, T* s, T r, int nx, const Kernel<T>& kernel, int* perm){//*
  int ny=nx;
  int nz=nx;

  T eps=machine_eps<T>()*64;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];

  std::vector<T> qp_x(nx), qw_x(nx);
  std::vector<T> qp_y(ny), qw_y(ny);
  std::vector<T> qp_z(nz), qw_z(nz);
  std::vector<T> p_x(nx*m);
  std::vector<T> p_y(ny*m);
  std::vector<T> p_z(nz*m);

  std::vector<T> x_;
  {
    x_.push_back(s[0]);
    x_.push_back(fabs(1.0-s[0])+s[0]);
    x_.push_back(fabs(1.0-s[1])+s[0]);
    x_.push_back(fabs(1.0+s[1])+s[0]);
    x_.push_back(fabs(1.0-s[2])+s[0]);
    x_.push_back(fabs(1.0+s[2])+s[0]);
    std::sort(x_.begin(),x_.end());
    for(int i=0;i<x_.size();i++){
      if(x_[i]<-1.0) x_[i]=-1.0;
      if(x_[i]> 1.0) x_[i]= 1.0;
    }

    std::vector<T> x_new;
    T x_jump=fabs(1.0-s[0]);
    if(fabs(1.0-s[1])>eps) x_jump=std::min(x_jump,(T)fabs(1.0-s[1]));
    if(fabs(1.0+s[1])>eps) x_jump=std::min(x_jump,(T)fabs(1.0+s[1]));
    if(fabs(1.0-s[2])>eps) x_jump=std::min(x_jump,(T)fabs(1.0-s[2]));
    if(fabs(1.0+s[2])>eps) x_jump=std::min(x_jump,(T)fabs(1.0+s[2]));
    for(int k=0; k<x_.size()-1; k++){
      T x0=x_[k];
      T x1=x_[k+1];

      T A0=0;
      T A1=0;
      { // A0
        T y0=s[1]-(x0-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        T y1=s[1]+(x0-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        T z0=s[2]-(x0-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        T z1=s[2]+(x0-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A0=(y1-y0)*(z1-z0);
      }
      { // A1
        T y0=s[1]-(x1-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
        T y1=s[1]+(x1-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
        T z0=s[2]-(x1-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
        T z1=s[2]+(x1-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;
        A1=(y1-y0)*(z1-z0);
      }
      T V=0.5*(A0+A1)*(x1-x0);
      if(V<eps) continue;

      if(!x_new.size()) x_new.push_back(x0);
      x_jump=std::max(x_jump,x0-s[0]);
      while(s[0]+x_jump*1.5<x1){
        x_new.push_back(s[0]+x_jump);
        x_jump*=2.0;
      }
      if(x_new.back()+eps<x1) x_new.push_back(x1);
    }
    assert(x_new.size()<30);
    x_.swap(x_new);
  }

  Vector<T> k_out(   ny*nz*k_dim,mem::aligned_new<T>(   ny*nz*k_dim),false);
  Vector<T> I0   (   ny*m *k_dim,mem::aligned_new<T>(   ny*m *k_dim),false);
  Vector<T> I1   (   m *m *k_dim,mem::aligned_new<T>(   m *m *k_dim),false);
  Vector<T> I2   (m *m *m *k_dim,mem::aligned_new<T>(m *m *m *k_dim),false); I2.SetZero();
  if(x_.size()>1)
  for(int k=0; k<x_.size()-1; k++){
    T x0=x_[k];
    T x1=x_[k+1];
    {
      std::vector<T> qp(nx);
      std::vector<T> qw(nx);
      quad_rule(nx,&qp[0],&qw[0]);
      for(int i=0; i<nx; i++)
        qp_x[i]=(x1-x0)*qp[i]/2.0+(x1+x0)/2.0;
      qw_x=qw;
    }
    cheb_poly(m-1,&qp_x[0],nx,&p_x[0]);

    for(int i=0; i<nx; i++){
      T y0=s[1]-(qp_x[i]-s[0]); if(y0<-1.0) y0=-1.0; if(y0> 1.0) y0= 1.0;
      T y1=s[1]+(qp_x[i]-s[0]); if(y1<-1.0) y1=-1.0; if(y1> 1.0) y1= 1.0;
      T z0=s[2]-(qp_x[i]-s[0]); if(z0<-1.0) z0=-1.0; if(z0> 1.0) z0= 1.0;
      T z1=s[2]+(qp_x[i]-s[0]); if(z1<-1.0) z1=-1.0; if(z1> 1.0) z1= 1.0;

      {
        std::vector<T> qp(ny);
        std::vector<T> qw(ny);
        quad_rule(ny,&qp[0],&qw[0]);
        for(int j=0; j<ny; j++)
          qp_y[j]=(y1-y0)*qp[j]/2.0+(y1+y0)/2.0;
        qw_y=qw;
      }
      {
        std::vector<T> qp(nz);
        std::vector<T> qw(nz);
        quad_rule(nz,&qp[0],&qw[0]);
        for(int j=0; j<nz; j++)
          qp_z[j]=(z1-z0)*qp[j]/2.0+(z1+z0)/2.0;
        qw_z=qw;
      }
      cheb_poly(m-1,&qp_y[0],ny,&p_y[0]);
      cheb_poly(m-1,&qp_z[0],nz,&p_z[0]);
      {
        T src[3]={0,0,0};
        std::vector<T> trg(ny*nz*3);
        for(int i0=0; i0<ny; i0++){
          size_t indx0=i0*nz*3;
          for(int i1=0; i1<nz; i1++){
            size_t indx1=indx0+i1*3;
            trg[indx1+perm[0]]=(s[0]-qp_x[i ])*r*0.5*perm[1];
            trg[indx1+perm[2]]=(s[1]-qp_y[i0])*r*0.5*perm[3];
            trg[indx1+perm[4]]=(s[2]-qp_z[i1])*r*0.5*perm[5];
          }
        }
        {
          Matrix<T> k_val(ny*nz*kernel.ker_dim[0],kernel.ker_dim[1]);
          kernel.BuildMatrix(&src[0],1,&trg[0],ny*nz,&k_val[0][0]);
          Matrix<T> k_val_t(kernel.ker_dim[1],ny*nz*kernel.ker_dim[0],&k_out[0], false);
          k_val_t=k_val.Transpose();
        }
        for(int kk=0; kk<k_dim; kk++){
          for(int i0=0; i0<ny; i0++){
            size_t indx=(kk*ny+i0)*nz;
            for(int i1=0; i1<nz; i1++){
              k_out[indx+i1] *= qw_y[i0]*qw_z[i1];
            }
          }
        }
      }

      I0.SetZero();
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<ny; i0++){
          size_t indx0=(kk*ny+i0)*nz;
          size_t indx1=(kk*ny+i0)* m;
          for(int i2=0; i2<m; i2++){
            for(int i1=0; i1<nz; i1++){
              I0[indx1+i2] += k_out[indx0+i1]*p_z[i2*nz+i1];
            }
          }
        }
      }

      I1.SetZero();
      for(int kk=0; kk<k_dim; kk++){
        for(int i2=0; i2<ny; i2++){
          size_t indx0=(kk*ny+i2)*m;
          for(int i0=0; i0<m; i0++){
            size_t indx1=(kk* m+i0)*m;
            T py=p_y[i0*ny+i2];
            for(int i1=0; i0+i1<m; i1++){
              I1[indx1+i1] += I0[indx0+i1]*py;
            }
          }
        }
      }

      T v=(x1-x0)*(y1-y0)*(z1-z0);
      for(int kk=0; kk<k_dim; kk++){
        for(int i0=0; i0<m; i0++){
          T px=p_x[i+i0*nx]*qw_x[i]*v;
          for(int i1=0; i0+i1<m; i1++){
            size_t indx0= (kk*m+i1)*m;
            size_t indx1=((kk*m+i0)*m+i1)*m;
            for(int i2=0; i0+i1+i2<m; i2++){
              I2[indx1+i2] += I1[indx0+i2]*px;
            }
          }
        }
      }
    }
  }
  for(int i=0;i<m*m*m*k_dim;i++)
    I2[i]=I2[i]*r*r*r/64.0;

  if(x_.size()>1)
  Profile::Add_FLOP(( 2*ny*nz*m*k_dim
                     +ny*m*(m+1)*k_dim
                     +2*m*(m+1)*k_dim
                     +m*(m+1)*(m+2)/3*k_dim)*nx*(x_.size()-1));

  std::vector<T> I2_(&I2[0], &I2[0]+I2.Dim());
  mem::aligned_delete<T>(&k_out[0]);
  mem::aligned_delete<T>(&I0   [0]);
  mem::aligned_delete<T>(&I1   [0]);
  mem::aligned_delete<T>(&I2   [0]);
  return I2_;
}

template <class T>
std::vector<T> integ(int m, T* s, T r, int n, const Kernel<T>& kernel){//*
  //Compute integrals over pyramids in all directions.
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  T s_[3];
  s_[0]=s[0]*2.0/r-1.0;
  s_[1]=s[1]*2.0/r-1.0;
  s_[2]=s[2]*2.0/r-1.0;

  T s1[3];
  int perm[6];
  std::vector<T> U_[6];

  s1[0]= s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[0]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[0];s1[1]=s_[1];s1[2]=s_[2];
  perm[0]= 0; perm[2]= 1; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[1]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]= s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[2]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[1];s1[1]=s_[0];s1[2]=s_[2];
  perm[0]= 1; perm[2]= 0; perm[4]= 2;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[3]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]= s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]= 1; perm[3]= 1; perm[5]= 1;
  U_[4]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  s1[0]=-s_[2];s1[1]=s_[0];s1[2]=s_[1];
  perm[0]= 2; perm[2]= 0; perm[4]= 1;
  perm[1]=-1; perm[3]= 1; perm[5]= 1;
  U_[5]=integ_pyramid<T>(m,s1,r,n,kernel,perm);

  std::vector<T> U; U.assign(m*m*m*k_dim,0);
  for(int kk=0; kk<k_dim; kk++){
    for(int i=0;i<m;i++){
      for(int j=0;j<m;j++){
        for(int k=0;k<m;k++){
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[0][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + j*m + i]+=U_[1][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[2][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + k*m*m + i*m + j]+=U_[3][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }

  for(int kk=0; kk<k_dim; kk++){
    for(int i=0; i<m; i++){
      for(int j=0; j<m; j++){
        for(int k=0; k<m; k++){
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[4][kk*m*m*m + i*m*m + j*m + k];
          U[kk*m*m*m + i*m*m + k*m + j]+=U_[5][kk*m*m*m + i*m*m + j*m + k]*(i%2?-1.0:1.0);
        }
      }
    }
  }
  return U;
}

template <class T>
std::vector<T> cheb_integ(int m, T* s_, T r_, const Kernel<T>& kernel){
  T eps=machine_eps<T>();
  T r=r_;
  T s[3]={s_[0],s_[1],s_[2]};
  int n=m+2;
  T err=1.0;
  int k_dim=kernel.ker_dim[0]*kernel.ker_dim[1];
  std::vector<T> U=integ<T>(m+1,s,r,n,kernel);
  std::vector<T> U_;
  while(err>eps*n){
    n=(int)round(n*1.3);
    if(n>300){
      std::cout<<"Cheb_Integ::Failed to converge.[";
      std::cout<<((double)err )<<",";
      std::cout<<((double)s[0])<<",";
      std::cout<<((double)s[1])<<",";
      std::cout<<((double)s[2])<<"]\n";
      break;
    }
    U_=integ<T>(m+1,s,r,n,kernel);
    err=0;
    for(int i=0;i<(m+1)*(m+1)*(m+1)*k_dim;i++)
      if(fabs(U[i]-U_[i])>err)
        err=fabs(U[i]-U_[i]);
    U=U_;
  }
  std::vector<T> U0(((m+1)*(m+2)*(m+3)*k_dim)/6);
  {
    int indx=0;
    const int* ker_dim=kernel.ker_dim;
    for(int l0=0;l0<ker_dim[0];l0++)
    for(int l1=0;l1<ker_dim[1];l1++)
    for(int i=0;i<=m;i++)
    for(int j=0;i+j<=m;j++)
    for(int k=0;i+j+k<=m;k++){
      U0[indx]=U[(k+(j+(i+(l0*ker_dim[1]+l1)*(m+1))*(m+1))*(m+1))];
      indx++;
    }
  }
  return U0;
}

template<typename T, void (*A)(T*, int, T*, int, T*, int, T*)>
Kernel<T> BuildKernel(const char* name, std::pair<int,int> k_dim,
    const Kernel<T>* k_s2m=NULL, const Kernel<T>* k_s2l=NULL, const Kernel<T>* k_s2t=NULL,
    const Kernel<T>* k_m2m=NULL, const Kernel<T>* k_m2l=NULL, const Kernel<T>* k_m2t=NULL,
    const Kernel<T>* k_l2l=NULL, const Kernel<T>* k_l2t=NULL, typename Kernel<T>::VolPoten vol_poten=NULL){
  Kernel<T> K(A, name, k_dim);
  K.k_s2m=k_s2m;
  K.k_s2l=k_s2l;
  K.k_s2t=k_s2t;
  K.k_m2m=k_m2m;
  K.k_m2l=k_m2l;
  K.k_m2t=k_m2t;
  K.k_l2l=k_l2l;
  K.k_l2t=k_l2t;
  K.vol_poten=vol_poten;
  return K;
}

template<class T>
struct LaplaceKernel{
  inline static const Kernel<T>& gradient();
};

template <class T>
Kernel<T>::Kernel(Ker_t poten, const char* name, std::pair<int,int> k_dim) {
  ker_dim[0]=k_dim.first;
  ker_dim[1]=k_dim.second;
  ker_poten=poten;
  ker_name=std::string(name);
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

template <class T>
void Kernel<T>::Initialize(bool verbose) const{
  if(init) return;
  init=true;
  T eps=1.0;
  while(eps+(T)1.0>1.0) eps*=0.5;
  T scal=1.0;
  if(ker_dim[0]*ker_dim[1]>0){
    Matrix<T> M_scal(ker_dim[0],ker_dim[1]);
    size_t N=1024;
    T eps_=N*eps;
    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*3);
    Matrix<T> M1(N,ker_dim[0]*ker_dim[1]);
    while(true){
      T abs_sum=0;
      for(size_t i=0;i<N/2;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=sqrtf(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*3+0]=x*scal;
        trg_coord1[i*3+1]=y*scal;
        trg_coord1[i*3+2]=z*scal;
      }
      for(size_t i=N/2;i<N;i++){
        T x,y,z,r;
        do{
          x=(drand48()-0.5);
          y=(drand48()-0.5);
          z=(drand48()-0.5);
          r=sqrtf(x*x+y*y+z*z);
        }while(r<0.25);
        trg_coord1[i*3+0]=x*1.0/scal;
        trg_coord1[i*3+1]=y*1.0/scal;
        trg_coord1[i*3+2]=z*1.0/scal;
      }
      for(size_t i=0;i<N;i++){
        BuildMatrix(&src_coord [          0], 1,
                    &trg_coord1[i*3], 1, &(M1[i][0]));
        for(size_t j=0;j<ker_dim[0]*ker_dim[1];j++){
          abs_sum+=fabs(M1[i][j]);
        }
      }
      if(abs_sum>sqrtf(eps) || scal<eps) break;
      scal=scal*0.5;
    }

    std::vector<T> trg_coord2(N*3);
    Matrix<T> M2(N,ker_dim[0]*ker_dim[1]);
    for(size_t i=0;i<N*3;i++){
      trg_coord2[i]=trg_coord1[i]*0.5;
    }
    for(size_t i=0;i<N;i++){
      BuildMatrix(&src_coord [          0], 1,
                  &trg_coord2[i*3], 1, &(M2[i][0]));
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
        M_scal[0][i]=log(s)/log(2.0);
        T err=sqrtf(0.5*(dot22/dot11)/(s*s)-0.5);
        if(err>eps_){
          scale_invar=false;
          M_scal[0][i]=0.0;
        }
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
      memcpy(&b[0][0],&M_scal[0][0],ker_dim[0]*ker_dim[1]*sizeof(T));
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
          if(fabs(src_scal[i0]+trg_scal[i1]-M_scal[i0][i1])>eps_){
            scale_invar=false;
          }
        }
      }
    }
    if(!scale_invar){
      src_scal.SetZero();
      trg_scal.SetZero();
    }
  }
  if(ker_dim[0]*ker_dim[1]>0){
    size_t N=1024;
    T eps_=N*eps;
    T src_coord[3]={0,0,0};
    std::vector<T> trg_coord1(N*3);
    std::vector<T> trg_coord2(N*3);
    for(size_t i=0;i<N/2;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sqrtf(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*3+0]=x*scal;
      trg_coord1[i*3+1]=y*scal;
      trg_coord1[i*3+2]=z*scal;
    }
    for(size_t i=N/2;i<N;i++){
      T x,y,z,r;
      do{
        x=(drand48()-0.5);
        y=(drand48()-0.5);
        z=(drand48()-0.5);
        r=sqrtf(x*x+y*y+z*z);
      }while(r<0.25);
      trg_coord1[i*3+0]=x*1.0/scal;
      trg_coord1[i*3+1]=y*1.0/scal;
      trg_coord1[i*3+2]=z*1.0/scal;
    }
    for(size_t p_type=0;p_type<C_Perm;p_type++){
      switch(p_type){
        case ReflecX:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]=-trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case ReflecY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]=-trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case ReflecZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]=-trg_coord1[i*3+2];
          }
          break;
        case SwapXY:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+1];
            trg_coord2[i*3+1]= trg_coord1[i*3+0];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
          break;
        case SwapXZ:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+2];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+0];
          }
          break;
        default:
          for(size_t i=0;i<N;i++){
            trg_coord2[i*3+0]= trg_coord1[i*3+0];
            trg_coord2[i*3+1]= trg_coord1[i*3+1];
            trg_coord2[i*3+2]= trg_coord1[i*3+2];
          }
      }
      Matrix<long long> M11, M22;
      {
        Matrix<T> M1(N,ker_dim[0]*ker_dim[1]); M1.SetZero();
        Matrix<T> M2(N,ker_dim[0]*ker_dim[1]); M2.SetZero();
        for(size_t i=0;i<N;i++){
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord1[i*3], 1, &(M1[i][0]));
          BuildMatrix(&src_coord [          0], 1,
                      &trg_coord2[i*3], 1, &(M2[i][0]));
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
            norm1[i]=sqrtf(dot11[i][i]);
            norm2[i]=sqrtf(dot22[i][i]);
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
              if(fabs(norm1[i]-norm1[j])<eps_ && fabs(fabs(dot11[i][j])-1.0)<eps_){
                M11[0][j]=(dot11[i][j]>0?flag:-flag);
              }
              if(fabs(norm1[i]-norm2[j])<eps_ && fabs(fabs(dot12[i][j])-1.0)<eps_){
                M22[0][j]=(dot12[i][j]>0?flag:-flag);
              }
            }
            flag++;
          }
        }
      }
      Matrix<long long> P1, P2;
      {
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
      {
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
      {
        Matrix<long long>& Pmat=P1;
        std::vector<Permutation<long long> >& Pvec=P1vec;
        Permutation<long long> P(Pmat.Dim(0));
        Vector<size_t>& perm=P.perm;
        perm.SetZero();
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }
        Vector<size_t> perm_tmp;
        while(true){
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
            size_t tmp=perm[i];
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
      {
        Matrix<long long>& Pmat=P2;
        std::vector<Permutation<long long> >& Pvec=P2vec;
        Permutation<long long> P(Pmat.Dim(0));
        Vector<size_t>& perm=P.perm;
        perm.SetZero();
        for(size_t i=0;i<P.Dim();i++)
        for(size_t j=0;j<P.Dim();j++){
          if(Pmat[i][j]){
            perm[i]=j;
            break;
          }
        }
        Vector<size_t> perm_tmp;
        while(true){
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
            size_t tmp=perm[i];
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
      {
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
      {
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
          {
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
  {
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

template <class T>
void Kernel<T>::BuildMatrix(T* r_src, int src_cnt, T* r_trg, int trg_cnt, T* k_out) const{
  memset(k_out, 0, src_cnt*ker_dim[0]*trg_cnt*ker_dim[1]*sizeof(T));
  for(int i=0;i<src_cnt;i++)
    for(int j=0;j<ker_dim[0];j++){
      std::vector<T> v_src(ker_dim[0],0);
      v_src[j]=1.0;
      ker_poten(&r_src[i*3], 1, &v_src[0], 1, r_trg, trg_cnt,
                &k_out[(i*ker_dim[0]+j)*trg_cnt*ker_dim[1]]);
    }
}


template <class Real_t, int SRC_DIM, int TRG_DIM, void (*uKernel)(Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&, Matrix<Real_t>&)>
void generic_kernel(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
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
  {
    size_t src_cnt_, trg_cnt_;
    src_cnt_=((src_cnt+VecLen-1)/VecLen)*VecLen;
    trg_cnt_=((trg_cnt+VecLen-1)/VecLen)*VecLen;
    size_t buff_size=src_cnt_*(3+SRC_DIM)+
                     trg_cnt_*(3+TRG_DIM);
    if(buff_size>STACK_BUFF_SIZE){
      buff=mem::aligned_new<Real_t>(buff_size);
    }
    Real_t* buff_ptr=buff;
    if(!buff_ptr){
      uintptr_t ptr=(uintptr_t)stack_buff;
      static uintptr_t     ALIGN_MINUS_ONE=MEM_ALIGN-1;
      static uintptr_t NOT_ALIGN_MINUS_ONE=~ALIGN_MINUS_ONE;
      ptr=((ptr+ALIGN_MINUS_ONE) & NOT_ALIGN_MINUS_ONE);
      buff_ptr=(Real_t*)ptr;
    }
    src_coord.ReInit(3, src_cnt_,buff_ptr,false);  buff_ptr+=3*src_cnt_;
    src_value.ReInit(  SRC_DIM, src_cnt_,buff_ptr,false);  buff_ptr+=  SRC_DIM*src_cnt_;
    trg_coord.ReInit(3, trg_cnt_,buff_ptr,false);  buff_ptr+=3*trg_cnt_;
    trg_value.ReInit(  TRG_DIM, trg_cnt_,buff_ptr,false);
    {
      size_t i=0;
      for(   ;i<src_cnt ;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=r_src[i*3+j];
        }
      }
      for(   ;i<src_cnt_;i++){
        for(size_t j=0;j<3;j++){
          src_coord[j][i]=0;
        }
      }
    }
    {
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
    {
      size_t i=0;
      for(   ;i<trg_cnt ;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=r_trg[i*3+j];
        }
      }
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<3;j++){
          trg_coord[j][i]=0;
        }
      }
    }
    {
      size_t i=0;
      for(   ;i<trg_cnt_;i++){
        for(size_t j=0;j<TRG_DIM;j++){
          trg_value[j][i]=0;
        }
      }
    }
  }
  uKernel(src_coord,src_value,trg_coord,trg_value);
  {
    for(size_t i=0;i<trg_cnt ;i++){
      for(size_t j=0;j<TRG_DIM;j++){
        v_trg[i*TRG_DIM+j]+=trg_value[j][i];
      }
    }
  }
  if(buff){
    mem::aligned_delete<Real_t>(buff);
  }
}

template <class Real_t, class Vec_t=Real_t>
void laplace_poten_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 1000
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);
  Real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = 1.0/(4*nwtn_scal*M_PI);
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
        Vec_t rinv=rsqrt_intrin2<Vec_t,Real_t>(r2);
        tv=add_intrin(tv,mul_intrin(rinv,sv));
      }
      Vec_t oofp=set_intrin<Vec_t,Real_t>(OOFP);
      tv=add_intrin(mul_intrin(tv,oofp),load_intrin<Vec_t>(&trg_value[0][t]));
      store_intrin(&trg_value[0][t],tv);
    }
  }
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*20);
  }
  #undef SRC_BLK
}

void laplace_poten(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
  generic_kernel<Real_t, 1, 1, laplace_poten_uKernel<Real_t,Vec_t> >(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

template <class Real_t, class Vec_t=Real_t>
void laplace_grad_uKernel(Matrix<Real_t>& src_coord, Matrix<Real_t>& src_value, Matrix<Real_t>& trg_coord, Matrix<Real_t>& trg_value){
  #define SRC_BLK 500
  size_t VecLen=sizeof(Vec_t)/sizeof(Real_t);
  Real_t nwtn_scal=1;
  for(int i=0;i<2;i++){
    nwtn_scal=2*nwtn_scal*nwtn_scal*nwtn_scal;
  }
  const Real_t OOFP = -1.0/(4*nwtn_scal*nwtn_scal*nwtn_scal*M_PI);
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
        Vec_t rinv=rsqrt_intrin2<Vec_t,Real_t>(r2);
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
  {
    Profile::Add_FLOP((long long)trg_cnt_*(long long)src_cnt_*27);
  }
  #undef SRC_BLK
}


void laplace_grad(Real_t* r_src, int src_cnt, Real_t* v_src, int dof, Real_t* r_trg, int trg_cnt, Real_t* v_trg){
  generic_kernel<Real_t, 1, 3, laplace_grad_uKernel<Real_t,Vec_t> >(r_src, src_cnt, v_src, dof, r_trg, trg_cnt, v_trg);
}

template<class T> const Kernel<T>& LaplaceKernel<T>::gradient(){
  static Kernel<T> potn_ker=BuildKernel<T, laplace_poten >("laplace"     , std::pair<int,int>(1,1));
  static Kernel<T> grad_ker=BuildKernel<T, laplace_grad >("laplace_grad", std::pair<int,int>(1,3),
      &potn_ker, &potn_ker, NULL, &potn_ker, &potn_ker, NULL, &potn_ker, NULL);
  return grad_ker;
}

}//end namespace

#endif //_PVFMM_FMM_KERNEL_HPP_

