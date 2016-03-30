#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include <iostream>

int main(int argc, char** argv) {
  __cilkrts_set_param("nworkers",argv[1]);
  std::cout << __cilkrts_get_nworkers() << std::endl;
}
