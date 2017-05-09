#include "coulomb_command.h"

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
  try{
    if(argc < 5){
      throw std::invalid_argument(std::string(argv[0]) + 
        " msm|ewald|test_msm parameter_file charge_file cordinate_file");
    }

    TimeLogger::print_current_time(std::cout);
    TimeLogger log(false);
    log.start();

    const int nthread = omp_get_max_threads();
    std::cout << "thread " << nthread << std::endl;
    std::cout << "sizeof(real) " << sizeof(real) << std::endl;

    const std::string method = argv[1];
    const std::string parameter_file = argv[2];
    const std::string charge_file = argv[3];
    const std::string crd_file = argv[4];

    std::map<std::string, std::string> parameter;

    Utility::read_parameter(parameter_file, parameter);

    System system;
    system.read_charge(charge_file);
    system.read_coordinate(crd_file);
    system.convert_unit_cell();

    system.print_parameter(std::cout);

    const int natom = system.get_atom_number();
    const real* q = system.get_charge();
    const real* r = system.get_coordinate();
    const real* L = system.get_unit_cell_length();


    Array::print_separator(std::cout);

    CoulombCommand command;

    command.set_parameter(parameter);

    if(method == "msm"){
      command.save_msm_potential_force(natom, q, r, L, std::cout);
    }
    else if(method == "ewald"){
      command.save_ewald_potential_force(natom, q, r, L, std::cout);
    }
    else if(method == "test_msm"){
      command.test_msm(natom, q, r, L, std::cout);
    }
    else{
      throw std::invalid_argument("Unknown method=" + method);
    }

    log.stop();
    Array::print_separator(std::cout);
    log.print_time("total");

    TimeLogger::print_current_time(std::cout);
  }
  catch(const std::exception& e){
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
