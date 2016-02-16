#include <kernel.hpp>

#ifndef _PVFMM_CHEB_UTILS_HPP_
#define _PVFMM_CHEB_UTILS_HPP_

namespace pvfmm{

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
      x_[i]=-pvfmm::cos<T>((T)(2.0*i+1.0)/(2.0*n)*const_pi<T>());
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
  {
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

  Vector<T> k_out(   ny*nz*k_dim,mem::aligned_new<T>(   ny*nz*k_dim,&mem_mgr),false);
  Vector<T> I0   (   ny*m *k_dim,mem::aligned_new<T>(   ny*m *k_dim,&mem_mgr),false);
  Vector<T> I1   (   m *m *k_dim,mem::aligned_new<T>(   m *m *k_dim,&mem_mgr),false);
  Vector<T> I2   (m *m *m *k_dim,mem::aligned_new<T>(m *m *m *k_dim,&mem_mgr),false); I2.SetZero();
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

}//end namespace

#endif //_PVFMM_CHEB_UTILS_HPP_

