#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////
void read(const std::string& file, std::vector<double>& x){
  std::ifstream in(file.c_str());

  if(!in){
    throw std::invalid_argument(file + " can not be opened.");
  }

  std::string line;

  while(std::getline(in, line)){
    x.push_back(std::stod(line));
  }
}

double get_relative_l2_error(const std::vector<double>& x,
  const std::vector<double>& y){
  double d2 = 0;
  double x2 = 0;

  for(int i = 0; i < x.size(); i++){
    double d = y[i] - x[i];
    d2 += d * d;
    x2 += x[i] * x[i];
  }

  return std::sqrt(d2 / x2);
}

double get_rmsre(const std::vector<double>& x,
  const std::vector<double>& y){
  double d2 = 0;

  for(int i = 0; i < x.size(); i++){
    double d = (y[i] - x[i]) / x[i];
    d2 += d * d;
  }

  return std::sqrt(d2 / x.size());
}

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]){
  try{
    const std::string x_file = argv[1];
    const std::string y_file = argv[2];

    std::vector<double> x, y;

    read(x_file, x);
    read(y_file, y);

    std::cout << get_relative_l2_error(x, y)
      << " " << get_rmsre(x, y)
      << std::endl;
  }
  catch(const std::exception& e){
    std::cerr << "ERROR: " << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

