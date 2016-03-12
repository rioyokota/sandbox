#ifndef _UTILS_
#define _UTILS_

const char* commandline_option(int argc, char** argv, const char* opt, const char* def_val, bool required, const char* err_msg){
  for(int i=0;i<argc;i++){
    if(!strcmp(argv[i],opt)){
      return argv[(i+1)%argc];
    }
  }
  return def_val;
}

#endif
