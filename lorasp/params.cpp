#include "params.h"
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>

params::params(char *address)
{

  std::ifstream param(address);
  std::ostringstream param_trimmed_out;
  std::string input_line;
  if(!param.good()) {
    std::cout << "params.in file does not exist." << std::endl;
    abort();
  }
  while (!param.eof())
    {
      std::getline(param,input_line);
      if ((int(input_line[0])!=0)&&(int(input_line[0])!=35)) param_trimmed_out<<input_line<<std::endl;
    }
  param.close();

  std::istringstream param_trimmed_in(param_trimmed_out.str());
  param_trimmed_in>>Input_Matrix_File_;
  param_trimmed_in>>treeLevelThreshold_;
  param_trimmed_in>>lowRankMethod_;
  param_trimmed_in>>cutOffMethod_;
  param_trimmed_in>>epsilon_;
  param_trimmed_in>>aPrioriRank_;
  param_trimmed_in>>rankCapFactor_;
  param_trimmed_in>>deployFactor_;
  param_trimmed_in>>gmresMaxIters_;
  param_trimmed_in>>gmresEpsilon_;
  param_trimmed_in>>gmresPC_;
  param_trimmed_in>>gmresVerbose_;
  param_trimmed_in>>ILUDropTol_;
  param_trimmed_in>>ILUFill_;
  param_trimmed_in>>normCols_;

}


params::params( char *matrixFile, int depth, int lrmeth, int cometh, double eps, int aprank, double rankCap, double depFac, int gmresMI, double gmresEps, int gmresPrec, double ILUDT, int ILUFil, bool normCols)
{

  Input_Matrix_File_ = std::string( matrixFile );
  treeLevelThreshold_ = depth;

  lowRankMethod_ = lrmeth;
  cutOffMethod_ = cometh;
  epsilon_ = eps;
  aPrioriRank_ = aprank;
  rankCapFactor_ = rankCap;
  deployFactor_ = depFac;
  gmresMaxIters_ = gmresMI;
  gmresEpsilon_ = gmresEps;
  gmresPC_ = gmresPrec;
  ILUDropTol_ = ILUDT;
  ILUFill_ = ILUFil;
  normCols_ = normCols;

  gmresVerbose_ = false;
}
