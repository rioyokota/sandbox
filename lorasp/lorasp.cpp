#include <iostream>
#include <stdlib.h>
#include "params.h"
#include "node.h"
#include "redNode.h"
#include "blackNode.h"
#include "superNode.h"
#include "tree.h"
#include "edge.h"
#include "time.h"
#include "gmres.h"
#include "Eigen/IterativeLinearSolvers"
#include "eyePC.h"
#include "diagPC.h"

int main(int argc, char *argv[])
{
  params *PARAM;
  if ( (argc != 2) && (argc !=15 ) )
    {
      std::cout<<"Please provide input files. e.g, ./LoRaSp param.in"<<std::endl;
      std::cout<<"Or provide parameters directly. e.g, ./LoRaSp voro4k/A.mtx voro4k/b.mtx 8 1e-2 1e-2 0 0"<<std::endl;
      std::cout<<"parameters are: path-to-matrix-file path-to-RHS-file tree-depth lowRankMeth cutOffMeth epsilon aPrioriRank rankCapFactor deployFactor gmresMaxIters gmresEpsilon gmresPC ILUDropTol ILUFill normCols"<<std::endl;
      exit(1);
    }
  else if ( argc == 2)
    {
      //! Creating the param object from params.in
      PARAM = new params( argv[1] );
    }
  else
    {
      //! Creating the param object from provided parameters
      PARAM = new params( argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atof(argv[5]), atoi(argv[6]), atof(argv[7]), atof(argv[8]), atoi(argv[9]), atof(argv[10]), atoi(argv[11]), atof(argv[12]), atoi(argv[13]), atoi(argv[14]) );
    }

  clock_t start, finish;
  start = clock();
  //! Creating the FMM tree with its root-node
  tree TREE(PARAM);
  finish = clock();
  std::cout<<" Creating FMM tree time = "<<double(finish-start)/CLOCKS_PER_SEC<<std::endl;
  TREE.assembleTime = double(finish-start)/CLOCKS_PER_SEC;

  start = clock();
  //! Create the columns to leaves map
  TREE.createCol2LeafMap();
  finish = clock();
  std::cout<<" Creating columns to leaves Map time = "<<double(finish-start)/CLOCKS_PER_SEC<<std::endl;

  start = clock();
  //! Create the Adjacency list for redNodes
  TREE.createAdjList();
  finish = clock();
  std::cout<<" Creating adjacency lists time = "<<double(finish-start)/CLOCKS_PER_SEC<<std::endl;

  start = clock();
  //! Create the edges between leaves of the tree
  TREE.createLeafEdges();
  finish = clock();
  std::cout<<" Creating Leaf Edges time = "<<double(finish-start)/CLOCKS_PER_SEC<<std::endl;

  start = clock();
  //! Dedicate memory, and set the RHS (and VARs) for the leaf nodes
  TREE.setLeafRHSVAR();
  finish = clock();
  std::cout<<" Set VAR for leaves time = "<<double(finish-start)/CLOCKS_PER_SEC<<std::endl;

  // Create a random solution and RHS
  VectorXd x = VectorXd::Random( TREE.n() );
  VectorXd b = ( *( TREE.Matrix() ) ) * x;
  // Initial guess vector
  VectorXd x0 = VectorXd::Zero( TREE.n() );
  std::cout<<" Artificial RHS and Solution created \n";

  switch ( PARAM->gmresPC() )
    {
    case 1: // i.e., Diag
      {
	start = clock();
	diagPC DIAGPC( TREE.Matrix() ) ;
	finish = clock();
	TREE.precondFactTime = double(finish-start)/CLOCKS_PER_SEC;
	gmres<diagPC> GMRES( TREE.Matrix(), &DIAGPC, &b, &x0, PARAM->gmresMaxIters(), PARAM->gmresEpsilon(), PARAM->gmresVerbose() );
	GMRES.solve( );

	x0 = x - *(GMRES.retVal());
	TREE.accuracy = x0.norm()/x.norm();
	x0 = b - (*TREE.Matrix()) * (*GMRES.retVal());
	TREE.residual = x0.norm()/b.norm();
	TREE.residuals = new VectorXd( GMRES.totalIters() );
	(*TREE.residuals) =  GMRES.residuals() -> segment( 0 , GMRES.totalIters() );
	TREE.gmresTotalTime = GMRES.totalTime();
	TREE.gmresTotalIters = GMRES.totalIters();
	break;
      }
    case 2: // i.e., ILU
      {
	start = clock();
	Eigen::IncompleteLUT<double> ILU( *(TREE.Matrix() ), PARAM->ILUDropTol(), PARAM->ILUFill() );
	finish = clock();
	TREE.precondFactTime = double(finish-start)/CLOCKS_PER_SEC;
	gmres<Eigen::IncompleteLUT<double>> GMRES( TREE.Matrix(), &ILU, &b, &x0, PARAM->gmresMaxIters(), PARAM->gmresEpsilon(), PARAM->gmresVerbose() );
	GMRES.solve( );

	x0 = x - *(GMRES.retVal());
	TREE.accuracy = x0.norm()/x.norm();
	x0 = b - (*TREE.Matrix()) * (*GMRES.retVal());
	TREE.residual = x0.norm()/b.norm();
	TREE.residuals = new VectorXd( GMRES.totalIters() );
	(*TREE.residuals) =  GMRES.residuals() -> segment( 0 , GMRES.totalIters() );
	TREE.gmresTotalTime = GMRES.totalTime();
	TREE.gmresTotalIters = GMRES.totalIters();
	break;
      }
    case 3: // i.e., H2
      {
	start = clock();
	TREE.factorize();
	finish = clock();
	TREE.precondFactTime = double(finish-start)/CLOCKS_PER_SEC;
	gmres<tree> GMRES( TREE.Matrix(), &TREE, &b, &x0, PARAM->gmresMaxIters(), PARAM->gmresEpsilon(), PARAM->gmresVerbose() );
	GMRES.solve( );

	x0 = x - *(GMRES.retVal());
	TREE.accuracy = x0.norm()/x.norm();
	x0 = b - (*TREE.Matrix()) * (*GMRES.retVal());
	TREE.residual = x0.norm()/b.norm();
	TREE.residuals = new VectorXd( GMRES.totalIters() );
	(*TREE.residuals) =  GMRES.residuals() -> segment( 0 , GMRES.totalIters() );
	TREE.gmresTotalTime = GMRES.totalTime();
	TREE.gmresTotalIters = GMRES.totalIters();
	break;
      }
    default: // i.e., Identity
      {
	start = clock();
	eyePC EYEPC;
	finish = clock();
	TREE.precondFactTime = double(finish-start)/CLOCKS_PER_SEC;
	gmres<eyePC> GMRES( TREE.Matrix(), &EYEPC, &b, &x0, PARAM->gmresMaxIters(), PARAM->gmresEpsilon(), PARAM->gmresVerbose() );
	GMRES.solve( );

	x0 = x - *(GMRES.retVal());
	TREE.accuracy = x0.norm()/x.norm();
	x0 = b - (*TREE.Matrix()) * (*GMRES.retVal());
	TREE.residual = x0.norm()/b.norm();
	TREE.residuals = new VectorXd( GMRES.totalIters() );
	(*TREE.residuals) =  GMRES.residuals() -> segment( 0 , GMRES.totalIters() );
	TREE.gmresTotalTime = GMRES.totalTime();
	TREE.gmresTotalIters = GMRES.totalIters();
	break;
      }

    }
  std::cout<<"\n*********GMRES CONVERGED***********\n\n";
  std::cout<<"Factorization TIME  = "<<TREE.precondFactTime<<std::endl;
  std::cout<<"GMRES TIME  = "<<TREE.gmresTotalTime<<std::endl;
  std::cout<<"GMRES NUM ITERS  = "<<TREE.gmresTotalIters<<std::endl;
  std::cout<<"GMRES RESIDUAL = "<<TREE.residual<<std::endl;
  std::cout<<"GMRES ACCURACY = "<<TREE.accuracy<<"\n\n\n";


  TREE.log("log.txt");
}
