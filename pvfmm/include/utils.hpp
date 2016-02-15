#ifndef _UTILS_
#define _UTILS_

#include <vector>
#include <cheb_utils.hpp>
#include <fmm_tree.hpp>

enum DistribType{
  UnifGrid,
  RandUnif,
  RandGaus,
  RandElps,
  RandSphr
};

template<typename FMMTree_t>
void CheckFMMOutput(FMMTree_t* mytree, const pvfmm::Kernel<Real_t>* mykernel, std::string t_name){
  if(mykernel==NULL) return;
  int np=omp_get_max_threads();
  int myrank=0, p=1;
  typedef typename FMMTree_t::FMMData FMM_Data_t;
  typedef typename FMMTree_t::FMMNode FMMNode_t;

  std::vector<Real_t> src_coord;
  std::vector<Real_t> src_value;
  FMMNode_t* n=static_cast<FMMNode_t*>(mytree->PreorderFirst());
  while(n!=NULL){
    if(n->IsLeaf() && !n->IsGhost()){
      pvfmm::Vector<Real_t>& coord_vec=n->src_coord;
      pvfmm::Vector<Real_t>& value_vec=n->src_value;
      for(size_t i=0;i<coord_vec.Dim();i++) src_coord.push_back(coord_vec[i]);
      for(size_t i=0;i<value_vec.Dim();i++) src_value.push_back(value_vec[i]);
    }
    n=static_cast<FMMNode_t*>(mytree->PreorderNxt(n));
  }
  long long src_cnt=src_coord.size()/3;
  long long val_cnt=src_value.size();
  if(src_cnt==0) return;
  int dof=val_cnt/src_cnt/mykernel->ker_dim[0];
  int trg_dof=dof*mykernel->ker_dim[1];
  std::vector<Real_t> trg_coord;
  std::vector<Real_t> trg_poten_fmm;
  long long trg_iter=0;
  size_t step_size=1+src_cnt*src_cnt*1e-9/p;
  n=static_cast<FMMNode_t*>(mytree->PreorderFirst());
  while(n!=NULL){
    if(n->IsLeaf() && !n->IsGhost()){
      pvfmm::Vector<Real_t>& coord_vec=n->trg_coord;
      pvfmm::Vector<Real_t>& poten_vec=n->trg_value;
      for(size_t i=0;i<coord_vec.Dim()/3          ;i++){
        if(trg_iter%step_size==0){
          for(int j=0;j<3        ;j++) trg_coord    .push_back(coord_vec[i*3        +j]);
          for(int j=0;j<trg_dof  ;j++) trg_poten_fmm.push_back(poten_vec[i*trg_dof  +j]);
        }
        trg_iter++;
      }
    }
    n=static_cast<FMMNode_t*>(mytree->PreorderNxt(n));
  }
  int trg_cnt=trg_coord.size()/3;
  if(trg_cnt==0) return;
  std::vector<Real_t> trg_poten_dir(trg_cnt*trg_dof ,0);
  pvfmm::Profile::Tic("N-Body Direct",false,1);
  #pragma omp parallel for
  for(int i=0;i<np;i++){
    size_t a=(i*trg_cnt)/np;
    size_t b=((i+1)*trg_cnt)/np;
    mykernel->ker_poten(&src_coord[0], src_cnt, &src_value[0], dof, &trg_coord[a*3], b-a, &trg_poten_dir[a*trg_dof  ],NULL);
  }
  pvfmm::Profile::Toc();
  {
    Real_t max_=0;
    Real_t max_err=0;
    for(size_t i=0;i<trg_poten_fmm.size();i++){
      Real_t err=fabs(trg_poten_dir[i]-trg_poten_fmm[i]);
      Real_t max=fabs(trg_poten_dir[i]);
      if(err>max_err) max_err=err;
      if(max>max_) max_=max;
    }
    if(!myrank){
#ifdef __VERBOSE__
      std::cout<<"Maximum Absolute Error ["<<t_name<<"] :  "<<std::scientific<<max_err<<'\n';
      std::cout<<"Maximum Relative Error ["<<t_name<<"] :  "<<std::scientific<<max_err/max_<<'\n';
#else
      std::cout<<"Error      : "<<std::scientific<<max_err/max_<<'\n';
#endif
    }
  }
}

template <class Real_t>
std::vector<Real_t> point_distrib(DistribType dist_type, size_t N){
  int np=1, myrank=0;
  static size_t seed=myrank+1; seed+=np;
  srand48(seed);

  std::vector<Real_t> coord;
  switch(dist_type){
  case UnifGrid:
    {
      size_t NN=(size_t)round(pow((double)N,1.0/3.0));
      size_t N_total=NN*NN*NN;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      for(size_t i=start;i<end;i++){
        coord.push_back(((Real_t)((i/  1    )%NN)+0.5)/NN);
        coord.push_back(((Real_t)((i/ NN    )%NN)+0.5)/NN);
        coord.push_back(((Real_t)((i/(NN*NN))%NN)+0.5)/NN);
      }
    }
    break;
  case RandUnif:
    {
      size_t N_total=N;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      size_t N_local=end-start;
      coord.resize(N_local*3);

      for(size_t i=0;i<N_local*3;i++)
        coord[i]=((Real_t)drand48());
    }
    break;
  case RandGaus:
    {
      Real_t e=2.7182818284590452;
      Real_t log_e=log(e);
      size_t N_total=N;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;

      for(size_t i=start*3;i<end*3;i++){
        Real_t y=-1;
        while(y<=0 || y>=1){
          Real_t r1=sqrt(-2*log(drand48())/log_e)*cos(2*M_PI*drand48());
          Real_t r2=pow(0.5,i*10/N_total);
          y=0.5+r1*r2;
        }
        coord.push_back(y);
      }
    }
    break;
  case RandElps:
    {
      size_t N_total=N;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      size_t N_local=end-start;
      coord.resize(N_local*3);

      const Real_t r=0.45;
      const Real_t center[3]={0.5,0.5,0.5};
      for(size_t i=0;i<N_local;i++){
        Real_t* y=&coord[i*3];
        Real_t phi=2*M_PI*drand48();
        Real_t theta=M_PI*drand48();
        y[0]=center[0]+0.25*r*sin(theta)*cos(phi);
        y[1]=center[1]+0.25*r*sin(theta)*sin(phi);
        y[2]=center[2]+r*cos(theta);
      }
    }
    break;
  case RandSphr:
    {
      size_t N_total=N;
      size_t start= myrank   *N_total/np;
      size_t end  =(myrank+1)*N_total/np;
      size_t N_local=end-start;
      coord.resize(N_local*3);

      const Real_t center[3]={0.5,0.5,0.5};
      for(size_t i=0;i<N_local;i++){
        Real_t* y=&coord[i*3];
        Real_t r=1;
        while(r>0.5 || r==0){
          y[0]=drand48(); y[1]=drand48(); y[2]=drand48();
          r=sqrt((y[0]-center[0])*(y[0]-center[0])
                +(y[1]-center[1])*(y[1]-center[1])
                +(y[2]-center[2])*(y[2]-center[2]));
          y[0]=center[0]+0.45*(y[0]-center[0])/r;
          y[1]=center[1]+0.45*(y[1]-center[1])/r;
          y[2]=center[2]+0.45*(y[2]-center[2])/r;
        }
      }
    }
    break;
  default:
    break;
  }
  return coord;
}

void commandline_option_start(int argc, char** argv, const char* help_text=NULL){
  char help[]="--help";
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],help)){
      if(help_text!=NULL) std::cout<<help_text<<'\n';
      std::cout<<"Usage:\n";
      std::cout<<"  "<<argv[0]<<" [options]\n\n";
    }
  }
}

const char* commandline_option(int argc, char** argv, const char* opt, const char* def_val, bool required, const char* err_msg){
  char help[]="--help";
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],help)){
      std::cout<<"        "<<err_msg<<'\n';
      return def_val;
    }
  }

  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],opt)){
      return argv[(i+1)%argc];
    }
  }
  if(required){
    std::cout<<"Missing: required option\n"<<"    "<<err_msg<<"\n\n";
    std::cout<<"To see usage options\n"<<"    "<<argv[0]<<" --help\n\n";
    exit(0);
  }
  return def_val;
}

void commandline_option_end(int argc, char** argv){
  char help[]="--help";
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],help)){
      std::cout<<"\n";
      exit(0);
    }
  }
}

#endif
