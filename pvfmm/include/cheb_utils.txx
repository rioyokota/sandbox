#include <omp.h>
#include <cmath>
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include <mat_utils.hpp>
#include <mem_mgr.hpp>
#include <matrix.hpp>
#include <profile.hpp>

namespace pvfmm{

template <class T>
T machine_eps(){
  T eps=1.0;
  while(eps+(T)1.0>1.0) eps*=0.5;
  return eps;
}

/**
 * \brief Returns the values of all chebyshev polynomials up to degree d,
 * evaluated at points in the input vector. Output format:
 * { T0[in[0]], ..., T0[in[n-1]], T1[in[0]], ..., T(d-1)[in[n-1]] }
 */
template <class T>
inline void cheb_poly(int d, const T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(pvfmm::fabs<T>(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(pvfmm::fabs<T>(in[i])<=1?1.0:0);
      out[i+n]=(pvfmm::fabs<T>(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(pvfmm::fabs<T>(in[j])<=1?in[j]:0);
      T y0=(pvfmm::fabs<T>(in[j])<=1?1.0:0);
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

/**
 * \brief Returns the sum of the absolute value of coeffecients of the highest
 * order polynomial as an estimate of error.
 */
template <class T>
T cheb_err(T* cheb_coeff, int deg, int dof){
  T err=0;
  int indx=0;

  for(int l=0;l<dof;l++)
  for(int i=0;i<=deg;i++)
  for(int j=0;i+j<=deg;j++)
  for(int k=0;i+j+k<=deg;k++){
    if(i+j+k==deg) err+=pvfmm::fabs<T>(cheb_coeff[indx]);
    indx++;
  }
  return err;
}


template<typename U1, typename U2>
struct SameType{
  bool operator()(){return false;}
};
template<typename U>
struct SameType<U, U>{
  bool operator()(){return true;}
};

template <class T>
inline void legn_poly(int d, T* in, int n, T* out){
  if(d==0){
    for(int i=0;i<n;i++)
      out[i]=(pvfmm::fabs<T>(in[i])<=1?1.0:0);
  }else if(d==1){
    for(int i=0;i<n;i++){
      out[i]=(pvfmm::fabs<T>(in[i])<=1?1.0:0);
      out[i+n]=(pvfmm::fabs<T>(in[i])<=1?in[i]:0);
    }
  }else{
    for(int j=0;j<n;j++){
      T x=(pvfmm::fabs<T>(in[j])<=1?in[j]:0);
      T y0=(pvfmm::fabs<T>(in[j])<=1?1.0:0);
      out[j]=y0;
      out[j+n]=x;

      T y1=x;
      T* y2=&out[2*n+j];
      for(int i=2;i<=d;i++){
        *y2=( (2*i-1)*x*y1-(i-1)*y0 )/i;
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

  { //Chebyshev quadrature nodes and weights
    for(int i=0;i<n;i++){
      x_[i]=-pvfmm::cos<T>((T)(2.0*i+1.0)/(2.0*n)*const_pi<T>());
      w_[i]=0;//pvfmm::sqrt<T>(1.0-x_[i]*x_[i])*const_pi<T>()/n;
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
  { // Set x_lst, w_lst
    x_lst[n]=x_;
    w_lst[n]=w_;
  }
  quad_rule(n, x, w);
}

template <class T>
std::vector<T> integ_pyramid(int m, T* s, T r, int nx, const Kernel<T>& kernel, int* perm){//*
  static mem::MemoryManager mem_mgr(16*1024*1024*sizeof(T));
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
  { //  Build stack along X-axis.
    x_.push_back(s[0]);
    x_.push_back(pvfmm::fabs<T>(1.0-s[0])+s[0]);
    x_.push_back(pvfmm::fabs<T>(1.0-s[1])+s[0]);
    x_.push_back(pvfmm::fabs<T>(1.0+s[1])+s[0]);
    x_.push_back(pvfmm::fabs<T>(1.0-s[2])+s[0]);
    x_.push_back(pvfmm::fabs<T>(1.0+s[2])+s[0]);
    std::sort(x_.begin(),x_.end());
    for(int i=0;i<x_.size();i++){
      if(x_[i]<-1.0) x_[i]=-1.0;
      if(x_[i]> 1.0) x_[i]= 1.0;
    }

    std::vector<T> x_new;
    T x_jump=pvfmm::fabs<T>(1.0-s[0]);
    if(pvfmm::fabs<T>(1.0-s[1])>eps) x_jump=std::min(x_jump,(T)pvfmm::fabs<T>(1.0-s[1]));
    if(pvfmm::fabs<T>(1.0+s[1])>eps) x_jump=std::min(x_jump,(T)pvfmm::fabs<T>(1.0+s[1]));
    if(pvfmm::fabs<T>(1.0-s[2])>eps) x_jump=std::min(x_jump,(T)pvfmm::fabs<T>(1.0-s[2]));
    if(pvfmm::fabs<T>(1.0+s[2])>eps) x_jump=std::min(x_jump,(T)pvfmm::fabs<T>(1.0+s[2]));
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

  Vector<T> k_out(   ny*nz*k_dim,mem::aligned_new<T>(   ny*nz*k_dim,&mem_mgr),false); //Output of kernel evaluation.
  Vector<T> I0   (   ny*m *k_dim,mem::aligned_new<T>(   ny*m *k_dim,&mem_mgr),false);
  Vector<T> I1   (   m *m *k_dim,mem::aligned_new<T>(   m *m *k_dim,&mem_mgr),false);
  Vector<T> I2   (m *m *m *k_dim,mem::aligned_new<T>(m *m *m *k_dim,&mem_mgr),false); I2.SetZero();
  if(x_.size()>1)
  for(int k=0; k<x_.size()-1; k++){
    T x0=x_[k];
    T x1=x_[k+1];

    { // Set qp_x
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

      { // Set qp_y
        std::vector<T> qp(ny);
        std::vector<T> qw(ny);
        quad_rule(ny,&qp[0],&qw[0]);
        for(int j=0; j<ny; j++)
          qp_y[j]=(y1-y0)*qp[j]/2.0+(y1+y0)/2.0;
        qw_y=qw;
      }
      { // Set qp_z
        std::vector<T> qp(nz);
        std::vector<T> qw(nz);
        quad_rule(nz,&qp[0],&qw[0]);
        for(int j=0; j<nz; j++)
          qp_z[j]=(z1-z0)*qp[j]/2.0+(z1+z0)/2.0;
        qw_z=qw;
      }
      cheb_poly(m-1,&qp_y[0],ny,&p_y[0]);
      cheb_poly(m-1,&qp_z[0],nz,&p_z[0]);
      { // k_out =  kernel x qw
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
  mem::aligned_delete<T>(&k_out[0],&mem_mgr);
  mem::aligned_delete<T>(&I0   [0],&mem_mgr);
  mem::aligned_delete<T>(&I1   [0],&mem_mgr);
  mem::aligned_delete<T>(&I2   [0],&mem_mgr);
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

/**
 * \brief
 * \param[in] r Length of the side of cubic region.
 */
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
      if(pvfmm::fabs<T>(U[i]-U_[i])>err)
        err=pvfmm::fabs<T>(U[i]-U_[i]);
    U=U_;
  }

  std::vector<T> U0(((m+1)*(m+2)*(m+3)*k_dim)/6);
  {// Rearrange data
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

template <class T>
std::vector<T> cheb_nodes(int deg, int dim){
  unsigned int d=deg+1;
  std::vector<T> x(d);
  for(int i=0;i<d;i++)
    x[i]=-pvfmm::cos<T>((i+(T)0.5)*const_pi<T>()/d)*0.5+0.5;
  if(dim==1) return x;

  unsigned int n1=pvfmm::pow<unsigned int>(d,dim);
  std::vector<T> y(n1*dim);
  for(int i=0;i<dim;i++){
    unsigned int n2=pvfmm::pow<unsigned int>(d,i);
    for(int j=0;j<n1;j++){
      y[j*dim+i]=x[(j/n2)%d];
    }
  }
  return y;
}


template <class T>
void cheb_diff(const Vector<T>& A, int deg, int diff_dim, Vector<T>& B, mem::MemoryManager* mem_mgr=NULL){
  size_t d=deg+1;

  // Precompute
  static Matrix<T> M;
  #pragma omp critical (CHEB_DIFF1)
  if(M.Dim(0)!=(size_t)d){
    M.Resize(d,d);
    for(size_t i=0;i<d;i++){
      for(size_t j=0;j<d;j++) M[j][i]=0;
      for(size_t j=1-(i%2);j<i;j=j+2){
        M[j][i]=2*i*2;
      }
      if(i%2==1) M[0][i]-=i*2;
    }
  }

  // Create work buffers
  size_t buff_size=A.Dim();
  T* buff=mem::aligned_new<T>(2*buff_size,mem_mgr);
  T* buff1=buff+buff_size*0;
  T* buff2=buff+buff_size*1;

  size_t n1=pvfmm::pow<unsigned int>(d,diff_dim);
  size_t n2=A.Dim()/(n1*d);

  for(size_t k=0;k<n2;k++){ // Rearrange A to make diff_dim the last array dimension
    Matrix<T> Mi(d,       n1,    &A[d*n1*k],false);
    Matrix<T> Mo(d,A.Dim()/d,&buff1[  n1*k],false);
    for(size_t i=0;i< d;i++)
    for(size_t j=0;j<n1;j++){
      Mo[i][j]=Mi[i][j];
    }
  }

  { // Apply M
    Matrix<T> Mi(d,A.Dim()/d,&buff1[0],false);
    Matrix<T> Mo(d,A.Dim()/d,&buff2[0],false);
    Matrix<T>::GEMM(Mo, M, Mi);
  }

  for(size_t k=0;k<n2;k++){ // Rearrange and write output to B
    Matrix<T> Mi(d,A.Dim()/d,&buff2[  n1*k],false);
    Matrix<T> Mo(d,       n1,    &B[d*n1*k],false);
    for(size_t i=0;i< d;i++)
    for(size_t j=0;j<n1;j++){
      Mo[i][j]=Mi[i][j];
    }
  }

  // Free memory
  mem::aligned_delete(buff,mem_mgr);
}

template <class T>
void cheb_grad(const Vector<T>& A, int deg, Vector<T>& B, mem::MemoryManager* mem_mgr){
  size_t dim=3;
  size_t d=(size_t)deg+1;
  size_t n_coeff =(d*(d+1)*(d+2))/6;
  size_t n_coeff_=pvfmm::pow<unsigned int>(d,dim);
  size_t dof=A.Dim()/n_coeff;

  // Create work buffers
  T* buff=mem::aligned_new<T>(2*n_coeff_*dof,mem_mgr);
  Vector<T> A_(n_coeff_*dof,buff+n_coeff_*dof*0,false); A_.SetZero();
  Vector<T> B_(n_coeff_*dof,buff+n_coeff_*dof*1,false); B_.SetZero();

  {// Rearrange data
    size_t indx=0;
    for(size_t l=0;l<dof;l++){
      for(size_t i=0;i<d;i++){
        for(size_t j=0;i+j<d;j++){
          T* A_ptr=&A_[(j+(i+l*d)*d)*d];
          for(size_t k=0;i+j+k<d;k++){
            A_ptr[k]=A[indx];
            indx++;
          }
        }
      }
    }
  }

  B.Resize(A.Dim()*dim);
  for(size_t q=0;q<dim;q++){
    // Compute derivative in direction q
    cheb_diff(A_,deg,q,B_);

    for(size_t l=0;l<dof;l++){// Rearrange data
      size_t indx=(q+l*dim)*n_coeff;
      for(size_t i=0;i<d;i++){
        for(size_t j=0;i+j<d;j++){
          T* B_ptr=&B_[(j+(i+l*d)*d)*d];
          for(size_t k=0;i+j+k<d;k++){
            B[indx]=B_ptr[k];
            indx++;
          }
        }
      }
    }
  }

  // Free memory
  mem::aligned_delete<T>(buff,mem_mgr);
}

template <class T>
void cheb_div(T* A_, int deg, T* B_){
  int dim=3;
  int d=deg+1;
  int n1 =pvfmm::pow<unsigned int>(d,dim);
  Vector<T> A(n1*dim); A.SetZero();
  Vector<T> B(n1    ); B.SetZero();

  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      A[k+(j+(i+l*d)*d)*d]=A_[indx];
      indx++;
    }
  }
  Matrix<T> MB(n1,1,&B[0],false);
  Matrix<T> MC(n1,1);
  for(int i=0;i<3;i++){
    {
      Vector<T> A_vec(n1,&A[n1*i],false);
      Vector<T> B_vec(n1,MC[0],false);
      cheb_diff(A_vec,deg,i,B_vec);
    }
    MB+=MC;
  }
  {// Rearrange data
    int indx=0;
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      B_[indx]=B[k+(j+i*d)*d];
      indx++;
    }
  }
}

template <class T>
void cheb_curl(T* A_, int deg, T* B_){
  int dim=3;
  int d=deg+1;
  int n1 =pvfmm::pow<unsigned int>(d,dim);
  Vector<T> A(n1*dim); A.SetZero();
  Vector<T> B(n1*dim); B.SetZero();

  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      A[k+(j+(i+l*d)*d)*d]=A_[indx];
      indx++;
    }
  }
  Matrix<T> MC1(n1,1);
  Matrix<T> MC2(n1,1);
  for(int i=0;i<3;i++){
    Matrix<T> MB(n1,1,&B[n1*i],false);
    int j1=(i+1)%3;
    int j2=(i+2)%3;
    {
      Vector<T> A1(n1,&A[n1*j1],false);
      Vector<T> A2(n1,&A[n1*j2],false);
      Vector<T> B1(n1,MC1[0],false);
      Vector<T> B2(n1,MC2[0],false);
      cheb_diff(A1,deg,j2,B2);
      cheb_diff(A1,deg,j1,B2);
    }
    MB=MC2;
    MB-=MC1;
  }
  {// Rearrange data
    int indx=0;
    for(int l=0;l<dim;l++)
    for(int i=0;i<d;i++)
    for(int j=0;i+j<d;j++)
    for(int k=0;i+j+k<d;k++){
      B_[indx]=B[k+(j+(i+l*d)*d)*d];
      indx++;
    }
  }
}

//TODO: Fix number of cheb_coeff to (d+1)*(d+2)*(d+3)/6 for the following functions.

template <class T>
void cheb_laplacian(T* A, int deg, T* B){
  int dim=3;
  int d=deg+1;
  int n1=pvfmm::pow<unsigned int>(d,dim);

  T* C1=mem::aligned_new<T>(n1);
  T* C2=mem::aligned_new<T>(n1);

  Matrix<T> M_(1,n1,C2,false);
  for(int i=0;i<3;i++){
    Matrix<T> M (1,n1,&B[n1*i],false);
    for(int j=0;j<n1;j++) M[0][j]=0;
    for(int j=0;j<3;j++){
      cheb_diff(&A[n1*i],deg,3,j,C1);
      cheb_diff( C1     ,deg,3,j,C2);
      M+=M_;
    }
  }

  mem::aligned_delete<T>(C1);
  mem::aligned_delete<T>(C2);
}

/*
 * \brief Computes image of the chebyshev interpolation along the specified axis.
 */
template <class T>
void cheb_img(T* A, T* B, int deg, int dir, bool neg_){
  int d=deg+1;
  int n1=pvfmm::pow<unsigned int>(d,3-dir);
  int n2=pvfmm::pow<unsigned int>(d,  dir);
  int indx;
  T sgn,neg;
  neg=(T)(neg_?-1.0:1.0);
  for(int i=0;i<n1;i++){
    indx=i%d;
    sgn=(T)(indx%2?-neg:neg);
    for(int j=0;j<n2;j++){
      B[i*n2+j]=sgn*A[i*n2+j];
    }
  }
}

}//end namespace
