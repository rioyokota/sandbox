#ifndef _UTILS_
#define _UTILS_

enum DistribType{
  UnifGrid,
  RandUnif,
  RandGaus,
  RandElps,
  RandSphr
};

std::vector<Real_t> point_distrib(size_t N){
  int np=1, myrank=0;
  static size_t seed=myrank+1; seed+=np;
  srand48(seed);
  std::vector<Real_t> coord;
  size_t NN=(size_t)round(pow((double)N,1.0/3.0));
  size_t N_total=NN*NN*NN;
  size_t start= myrank   *N_total/np;
  size_t end  =(myrank+1)*N_total/np;
  for(size_t i=start;i<end;i++){
    coord.push_back(((Real_t)((i/  1    )%NN)+0.5)/NN);
    coord.push_back(((Real_t)((i/ NN    )%NN)+0.5)/NN);
    coord.push_back(((Real_t)((i/(NN*NN))%NN)+0.5)/NN);
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
