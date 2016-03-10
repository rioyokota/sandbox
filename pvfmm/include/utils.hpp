#ifndef _UTILS_
#define _UTILS_

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

#endif
