#include "tree.h"
#include "redNode.h"
#include "blackNode.h"
#include "superNode.h"
#include "params.h"
#include "edge.h"
#include "Eigen/Sparse"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <cmath>
#include "time.h"

// Constructor
tree::tree(params* par)
{

  // param_ get value
  param_ = par;

  /*******************************************************/
  /* READ MATRIX AND RHS FROM FILE AND STORE AS TRIPLETS */
  /*******************************************************/

  // Read the input matrix data
  /* We will store input data as triplets */

  std::ifstream matrix_file( (char*) (param_->Input_Matrix_File().c_str()) );
  if(!matrix_file.good()) {
    std::cout << "A.mtx file does not exist." << std::endl;
    abort();
  }

  //number of rows, columns, and non-zero elements in the matrix
  int nrows , ncols;

  std::string first_line;
  std::getline(matrix_file,first_line);
  bool symmetric = false;
  if ( first_line[0]=='%' ) // i.e., file has a direction header
    {
      std::size_t found = first_line.find("real symmetric");
      if ( found != std::string::npos ) //i.e., it's symmetric
	{
	  symmetric = true;
	  std::cout<<"SYMMETRIC!\n";
	}
      matrix_file >> nrows >> ncols >> nnz;
    }
  else
    {
      std::istringstream first_line_in(first_line);
      first_line_in >> nrows >> ncols >> nnz;
    }

  n_ = nrows;

  std::cout<<"numRows = "<<nrows<<" , numCols = "<<ncols<<" , NNZ = "<<nnz<<"\n";

  if ( nrows != ncols )
    {
      std::cout<<" This code does not suppot non-square matrices!"<<std::endl;
      exit(1);
    }

  // The i,j indecies of an entry
  int index_i, index_j;

  // The value of an entry
  double value_ij;

  // Reserve memory for the triplet list vector
  TripletList_.reserve(n_);
  TripletBoolList_.reserve(n_);


  for ( int i = 0; i < nnz; i++ )
    {
      matrix_file>>index_i>>index_j>>value_ij;
      //  Changing index from 1,2,.. to 0,1,..
      TripletList_.push_back( TRIPLET( index_i-1, index_j-1, value_ij ) );
      if ( index_i != index_j )
	{
	  TripletBoolList_.push_back( TRIPLETBOOL( index_i-1 , index_j-1 , true));
	  // FOR SYMMETRIC MATRIX
	  if ( symmetric )
	    {
	      TripletList_.push_back( TRIPLET( index_j-1, index_i-1, value_ij ) );
	      TripletBoolList_.push_back( TRIPLETBOOL( index_j-1 , index_i-1 , true));
	    }
	}
    }

  matrix_file.close();

  // Init. RHS to all zeros
  RHS_ = new VectorXd( nrows );
  *RHS_ = VectorXd::Zero( nrows );

  // Allocate memory to the solution vector
  VAR_ = new VectorXd( n_ );


  // Assigne memory for the full matrix
  matrix_ = new spMat(n_,n_);
  symb_matrix_ = new spMatBool(n_,n_);
  permuted_matrix_ = new spMat(n_,n_);
  permuted_symb_matrix_ = new spMatBool(n_,n_);

  // Make a SparseMatrix from triplets
  matrix_->setFromTriplets( TripletList_.begin(),TripletList_.end() );
  symb_matrix_->setFromTriplets( TripletBoolList_.begin(),TripletBoolList_.end() );

  // Make the matrix in regular compressed sparse form
  //normalize_rows( matrix_ );
  matrix_ -> makeCompressed();
  symb_matrix_ -> makeCompressed();

  // Max max per column = 1
  if ( param_ -> normCols() )
    {
      normalize_cols( matrix_ );
    }

  // Reserve memory for redNode, superNode, frobNorm, and totalSize lists:
  redNodeList_.reserve( param_->treeLevelThreshold() + 1 );
  superNodeList_.reserve( param_->treeLevelThreshold() + 1 );
  frobNorms.reserve( param_->treeLevelThreshold() + 1 );
  totalSizes.reserve( param_->treeLevelThreshold() + 1 );
  for ( unsigned int l = 0; l <= param_->treeLevelThreshold(); l++ )
    {
      redNodeStrList newRedLevel;
      redNodeList_.push_back( newRedLevel );
      redNodeList_.back().reserve( 1<<l );

      superNodeStrList newSuperLevel;
      superNodeList_.push_back( newSuperLevel );
      superNodeList_.back().reserve( 1<<l );

      frobNorms.push_back(0);
      totalSizes.push_back(0);
    }

  /*******************************************************/
  /*                  Create the tree                    */
  /*******************************************************/

  //! Constructing the root
  /*!
    The root of the tree is a redNode.
    We need to pass the range of columns that belong to each node.
    The tree will be created in a BFS order.
    The order of things need to be done:
    - create the root
    - When a redNode born it will be added to the redNodelist
    - for the last created level do the permuation
    - for the last created level decide if further subdividing is required (based on threshold_level)
    - if yes for all redNodes in the last level call createBlacknode()
  */

  // Allocate memory to the pemuration vecotr
  permutationVector = new VectorXi(n_);

  //Allocate memory to the permutation matrix
  permutationMatrix = new permMat(n_);

  // Define the root of tree
  // We arbitrarily assume it is a left child (which doesn not matter)
  root_ = new redNode( 0, this, 0, 0, n_-1 );

  // Since the 0'th level does not have any actual superNode, we just create an empty list
  superNodeStrList NewLevel;
  superNodeList_.push_back(NewLevel);

  // std::cout<<"The original Symb Matrix = "<<*symb_matrix_<<std::endl;

  // loop over levels, create them, and perform permutations
  for ( unsigned int l = 0 ; l < param_->treeLevelThreshold(); l++ )
    {
      // loop over redNodes of level l, and create their black list.
      // this will be automatically followed by creating the redNode children.
      for ( unsigned int i = 0; i < redNodeList_[l].size(); i++ )
	{
	  redNodeList_[l][i]->createBlackNode();
	}
      // At this point new redNodes are added to the tree.
      // Hence, we need to permute the matrix accordingly
      permuteMatrix();
      // std::cout<<"The Symb Matrix after creating level "<<l+1<<" = "<<*symb_matrix_<<std::endl;
    }

  //! set the global frob norm = 0 initially
  globFrobNorm = 0;
  globSize = 0;

  //! Initially no node is eliminated
  count = 0;

  //! Initially no padding and no large pivot is assumed
  padding = false;
  largePivot = false;

}


// Adding a new red node to the list of red nodes of the tree
void tree::addRedNode( unsigned int l, redNode* ptr)
{

  // Check if this is the first redNode of a new level
  /*
  if ( l >= redNodeList_.size() )
    {
      redNodeStrList NewLevel;
      redNodeList_.push_back(NewLevel);
    }
  */

  // Push the ptr to redNode in the corresponding level
  redNodeList_[l].push_back(ptr);
}

// Adding a new red node to the list of red nodes of the tree
void tree::addSuperNode( unsigned int l, superNode* ptr)
{
  // Check if this is the first redNode of a new level
  /*
  if ( l >= superNodeList_.size() )
    {
      superNodeStrList NewLevel;
      superNodeList_.push_back(NewLevel);
      // std::cout<<" this happend with l = "<<l<<"\n";

      // Also push back 0 to FrobNorm list of Frob. norms
      frobNorms.push_back(0);
      totalSizes.push_back(0);
    }
  */

  // Push the ptr to redNode in the corresponding level
  superNodeList_[l].push_back(ptr);
}


// Use permuteationVector and permute the matrix
// Note that the current version of Eigen does not support in-place permutation
void tree::permuteMatrix()
{
  (*permutationMatrix) = permMat(*permutationVector);

  *permuted_matrix_ = matrix_->transpose();
  *matrix_ = (*permutationMatrix) * (*permuted_matrix_);
  *permuted_matrix_ = matrix_->transpose();
  *matrix_ = (*permutationMatrix) * (*permuted_matrix_);

  *permuted_symb_matrix_ = symb_matrix_->transpose();
  *symb_matrix_ = (*permutationMatrix) * (*permuted_symb_matrix_);
  *permuted_symb_matrix_ = symb_matrix_->transpose();
  *symb_matrix_ = (*permutationMatrix) * (*permuted_symb_matrix_);

  (*RHS_) = (*permutationMatrix) * (*RHS_);

}

void tree::createCol2LeafMap()
{

  // Index of the first and last columns corresponding to each leaf node
  int first, last;

  // Pointer to the node of interest
  redNode* Leaf;
  // loop over all leaf redNodes
  for ( unsigned int i = 0 ; i < redNodeList_[maxLevel()].size(); i++ )
    {
      Leaf = redNodeList_[maxLevel()][i];
      first = Leaf->IndexFirst();
      last = Leaf->IndexLast();
      if ( !Leaf->isEmpty() ) //i.e., if the leaf is not empty
	{
	  col2Leaf_.insert( std::pair<int,redNode*> (first, Leaf ) );
	}
    }

}

void tree::createAdjList()
{

  // Index of the first and last colum corresponding to each leaf node
  int first, last;

  // Pointer to the node of interest, and its adjacent node
  redNode* Leaf;
  redNode* AdjLeaf;

  // Iteration variable to work with col2Leaf map
  std::map<int,redNode*>::iterator it;

  // First create the adjacency list for the leaf nodes
    for ( unsigned int i = 0 ; i < redNodeList_[maxLevel()].size(); i++ )
    {
      Leaf = redNodeList_[maxLevel()][i];
      first = Leaf->IndexFirst();
      last = Leaf->IndexLast();

      Leaf -> AdjList() -> insert(Leaf); // Make sure to consider self interaction

      if ( !Leaf -> isEmpty() ) //i.e., if the leaf is not empty
	{
	  // Extract the list of rows that have interaction with a column in this leaf node
	  for ( int j = SymbMatrix()->outerIndexPtr()[first]; j < SymbMatrix()->outerIndexPtr()[last+1]; j++ )
	    {
	      it = col2Leaf_.upper_bound(SymbMatrix()->innerIndexPtr()[j]); //This give us the iterator for the leaf after the adjacent leaf in the map
	      it--; // This is the iterator for the adjacent node
	      AdjLeaf = it->second;
	      Leaf -> AdjList() -> insert(AdjLeaf); // Add newly found adjacent list to the list of adjacent lists
	    }
	}
    }

    // Pointer to the redNode of interest, and the left and right children of its black child
    redNode* myRedNode;
    redNode* leftRedChild;
    redNode* rightRedChild;

    // The adjacency set of my (grand child)
    std::set<redNode*> *childAdjSet;

    // Now, from bottom to top we create the adjacency list of all levels redNodes
    for ( int i = maxLevel()-1; i >= 0; i-- )
      {
	for ( unsigned int j = 0; j < redNodeList_[i].size(); j++)
	  {
	    myRedNode = redNodeList_[i][j];
	    leftRedChild = myRedNode -> child() -> leftChild();
	    rightRedChild = myRedNode -> child() -> rightChild();

	    // Go through (grand) parent of adjacent nodes of my (grand) children, and add them to my adjacent list

	    // Left (grand) child
	    childAdjSet = leftRedChild -> AdjList();
	    for ( std::set<redNode*>::iterator it = childAdjSet -> begin(); it != childAdjSet -> end(); ++it )
	      {
		myRedNode -> AdjList() -> insert( ( (*it) -> parent() ) -> parent() );
	      }

	    // Right (grand) child
	    childAdjSet = rightRedChild -> AdjList();
	    for ( std::set<redNode*>::iterator it = childAdjSet->begin(); it != childAdjSet->end(); ++it )
	      {
		myRedNode -> AdjList() -> insert( ( (*it) -> parent() ) -> parent() );
	      }
	  }
      }
}

// Create the edges between leaves of the tree
void tree::createLeafEdges()
{
  // The set of adjacent nodes to a leaf
  std::set<redNode*> *leafAdjSet;

  // go through all leaves
  for ( unsigned int i = 0; i < redNodeList_.back().size(); i++ )
    {
      // pointer to a leaf
      redNode* leaf = redNodeList_.back()[i];
      // pointer to its adjacency list
      leafAdjSet = leaf -> AdjList();
      // location of columns corresponding to this leaf
      int colFirst = leaf->IndexFirst();
      int colWidth = leaf->IndexLast() - colFirst + 1;

      // Go through adjacency list of each node
      for ( std::set<redNode*>::iterator it = leafAdjSet->begin(); it != leafAdjSet->end(); ++it )
	{
	  // location of rows corresponding to the adjacent leaf
	  int rowFirst = (*it)->IndexFirst();
	  int rowWidth = (*it)->IndexLast() - rowFirst + 1;

	  // Creating the edge and add it to in/out edges list of two leaves
	  leaf->outEdges.push_back(new edge(leaf, *it));
	  (*it)->inEdges.push_back(leaf->outEdges.back());

	  // Create the interaction matrix and put that on the edge
	  leaf -> outEdges.back() -> matrix = new densMat( matrix_ -> block(rowFirst, colFirst, rowWidth, colWidth) );
	}

    }

}

// Create superNodes at level [l] of the tree
double tree::createSuperNodes( unsigned int l )
{
  //std:: cout << " list of level " << l << " superNodes:" << std::endl;
  // Go through all redNodes at level [l-1], then call the mergeChildren function for their black child
  double time_elapsed = 0;
  for ( unsigned int i = 0; i < redNodeList_[l-1].size(); i++ )
    {
      time_elapsed += redNodeList_[l-1][i] -> child() -> mergeChildren();
      //std::cout << " superNode = " << redNodeList_[l-1][i] -> child() -> superChild() << "  , blackParent = " << redNodeList_[l-1][i] -> child() << "  , redParent = " << redNodeList_[l-1][i] << std::endl;
    }
  return time_elapsed;
  //std::cout<<std::endl;
}

// Eliminate nodes at level [l] of the tree
timeTuple4 tree::eliminate( unsigned int l )
{
  timeTuple4 times;
  times[0] = 0;
  times[1] = 0;
  times[2] = 0;
  times[3] = 0;

  timeTuple2 comp_time;
  timeTuple2 sc_time;

  // std::cout << "Start eliminating level " << l << std::endl;
  // Go through all superNodes at level [l], compress and apply schurComp.
  for ( unsigned int i = 0; i < superNodeList_[l].size(); i++ )
    {
      // std::cout << "  Start eliminating node " << i << std::endl;
      comp_time = superNodeList_[l][i] -> compress();
      times[0] += comp_time[0];
      times[1] += comp_time[1];

      // std::cout << "    Compression done! "<< std::endl;
      sc_time = superNodeList_[l][i] -> schurComp();
      times[2] += sc_time[0];
      times[3] += sc_time[1];
      // std::cout << "    Schur Comp. done! "<< std::endl;
    }
  // std::cout<<std::endl;
  return times;
}

// solveU blackNodes and superNodes at level[l], (superNodes solution split to redNodes solution)
void tree::solveU( unsigned int l )
{

  for ( unsigned int i = superNodeList_[l].size() ; i > 0; i-- )
    {
      // Solve for the blackNode first
      superNodeList_[l][i-1] -> parent() -> solveU();
      //      std::cout<<" blackNode solved! \n";

      // Then solve for the superNode
      superNodeList_[l][i-1] -> solveU();
      //      std::cout<<" superNode solved! \n";
    }

  for ( unsigned int i = superNodeList_[l].size(); i > 0; i-- )
    {
      // Then split solution to the redNodes
      superNodeList_[l][i-1] -> splitVAR();
      //      std::cout<<" redNodes solved! \n";
    }
}

// solveL blackNodes and superNodes at level[l], (first RHS of redNodes merge to create RHS of superNode)
void tree::solveL( unsigned int l )
{
  // Merge RHS of redNodes to RHS of the superNode
  for ( unsigned int i = 0; i < superNodeList_[l].size(); i++ )
    {
      superNodeList_[l][i] -> parent() -> mergeRHS();
    }
  for ( unsigned int i = 0; i < superNodeList_[l].size(); i++ )
    {
      superNodeList_[l][i] -> solveL();
      // std::cout<<"order (superNode) = " << superNodeList_[l][i]->order<<"    ";
      superNodeList_[l][i] -> parent() -> solveL();
      // std::cout<<"order (blackNode) = " << superNodeList_[l][i]->parent()->order<<"   ";
    }
}

// Set leaf level RHS
void tree::setRHS( VectorXd& rhs_ )
{

  // first set the RHS of the leaf redNodes
  setLeafRHS( rhs_ );

  // now set the RHS of all other red and black nodes zero
  for ( unsigned int l = 0; l < redNodeList_.size()-1; l++)
    {
      for ( unsigned int i = 0; i < redNodeList_[l].size(); i++ )
	{
	  // pointer to a red node
	  redNode* rNode = redNodeList_[l][i];

	  if ( rNode->m() > 0 )
	    {
	      // make RHS of the redNode 0
	      //*( rNode -> RHS() ) = VectorXd::Zero( rNode->m() );
	      rNode -> RHS() -> setZero();
	    }

	  if ( rNode->child()->m() > 0 )
	    {
	      // make RHS of its black child zero
	      //( rNode -> child() -> RHS() ) = VectorXd::Zero( rNode -> child() -> m() );
	      rNode -> child() -> RHS() -> setZero();
	    }

	}
    }
}


// Set leaf level RHS
void tree::setLeafRHS( VectorXd& rhs_ )
{

  // go through all leaves
  for ( unsigned int i = 0; i < redNodeList_.back().size(); i++ )
    {
      // pointer to a leaf
      redNode* leaf = redNodeList_.back()[i];

      // location of columns/rows corresponding to this leaf
      int colFirst = leaf->IndexFirst();
      int colWidth = leaf->IndexLast() - colFirst + 1;

      // Assign the RHS
      *( leaf -> RHS() ) = rhs_.segment( colFirst, colWidth );

    }
}

// set leaf level VAR
void tree::setLeafRHSVAR()
{

  // go through all leaves
  for ( unsigned int i = 0; i < redNodeList_.back().size(); i++ )
    {
      // pointer to a leaf
      redNode* leaf = redNodeList_.back()[i];

      // location of columns/rows corresponding to this leaf
      int colFirst = leaf->IndexFirst();
      int colWidth = leaf->IndexLast() - colFirst + 1;

      // Allocate memory for VAR
      leaf -> VAR( new VectorXd( colWidth ) );

      // Allocate memory RHS
      leaf -> RHS( new VectorXd( colWidth ) );

      *( leaf -> RHS() ) = RHS_ -> segment( colFirst, colWidth );
    }
}

// collect the solution of all leaves
VectorXd& tree::computeSolution()
{

  // number of filled
  int numFilled = 0;

  // Go through all leaves, and concatenate their solutions
  for ( unsigned int i = 0; i < redNodeList_.back().size(); i++ )
    {
      node* Leaf = redNodeList_.back()[i];
      if ( Leaf->n() > 0 ) // i.e., if this leaf has any unknown
	{
	  VAR_ -> segment( numFilled, Leaf->n() ) = *( Leaf->VAR() );
	  numFilled += Leaf->n();
	}
    }
  return *VAR_;
}

// Compute mean, min, and max size of redNodes at each level
void tree::setRanks()
{
  // Go through every level
  for ( unsigned int l = 0; l < redNodeList_.size(); l++ )
    {
      double mean = 0;
      int minimum = 1e6;
      int maximum = -1e6;
      // Go through every redNode
      for ( unsigned int i = 0; i < redNodeList_[l].size(); i++ )
	{
	  int rank = redNodeList_[l][i]->n();
	  mean += rank;
	  minimum = std::min( minimum, rank );
	  maximum = std::max( maximum, rank );
	}
      meanRanks_.push_back( mean / redNodeList_[l].size() );
      minRanks_.push_back( minimum  );
      maxRanks_.push_back( maximum );
    }
}

// log informations:
// matrix_size nnz nnz_RHS epsilon epsilonR method methodR AssembleTime H2_solve_time SparseLU_sole_time accuracy depth meanRanks[] minRanks[] maxRanks[]
void tree::log( std::string fileName )
{
  std::ofstream output;
  output.open( fileName.c_str(), std::fstream::out|std::fstream::app );

  std::ofstream output2;
  output2.open( "rlog.txt", std::fstream::out|std::fstream::app );

  output<<"matrixSize= "<<n_<<"\n";
  output<<"nnzMatrix= "<<nnz<<"\n";
  output<<"Padding happened? "<<padding<<"\n";
  output<<"LargePivot happened? "<<largePivot<<"\n";
  output<<"low-rank method= "<<param_->lowRankMethod()<<"\n";
  output<<"epsilon= "<<param_->epsilon()<<"\n";
  output<<"cut-off method= "<<param_->cutOffMethod()<<"\n";
  output<<"a priori rank= "<<param_->aPrioriRank()<<"\n";
  output<<"Rank cap factor= "<<param_->rankCapFactor()<<"\n";
  output<<"rSVD deploy factor= "<<param_->deployFactor()<<"\n";
  output<<"assembleTime= "<<assembleTime<<"\n";
  output<<"PrecondFactTime= "<<precondFactTime<<"\n";
  output<<"GMRES totalTime= "<<gmresTotalTime<<"\n";
  output<<"treeDepth= "<<param_->treeLevelThreshold()<<"\n";
  output<<"GMRES epsilon = "<<param_->gmresEpsilon()<<"\n";
  output<<"GMRES Preconditinoer = "<<param_->gmresPC()<<"\n";
  output<<"ILU DropTol = "<<param_->ILUDropTol()<<"\n";
  output<<"ILU Fill = "<<param_->ILUFill()<<"\n";
  output<<"GMRES final (actual) residual = "<<residual<<"\n";
  output<<"GMRES final accuracy = "<<accuracy<<"\n";
  output<<"GMRES maxIters = "<<param_->gmresMaxIters()<<"\n";
  output<<"GMRES totalIters= "<<gmresTotalIters<<"\n";
  output<<"\n\n";

  output.close();

  // Store the data-only version
  output2<<n_<<" ";
  output2<<nnz<<" ";
  output2<<padding<<" ";
  output2<<largePivot<<" ";
  output2<<param_->lowRankMethod()<<" ";
  output2<<param_->epsilon()<<" ";
  output2<<param_->cutOffMethod()<<" ";
  output2<<param_->aPrioriRank()<<" ";
  output2<<param_->rankCapFactor()<<" ";
  output2<<param_->deployFactor()<<" ";
  output2<<assembleTime<<" ";
  output2<<precondFactTime<<" ";
  output2<<gmresTotalTime<<" ";
  output2<<param_->treeLevelThreshold()<<" ";
  output2<<param_->gmresEpsilon()<<" ";
  output2<<param_->gmresPC()<<" ";
  output2<<param_->ILUDropTol()<<" ";
  output2<<param_->ILUFill()<<" ";
  output2<<residual<<" ";
  output2<<accuracy<<" ";
  output2<<param_->gmresMaxIters()<<" ";
  output2<<gmresTotalIters<<" ";
  for ( int i = 0; i < gmresTotalIters; i++ )
    {
      output2<<(*residuals)(i)<<" ";
    }
  output2<<"\n";
  output2.close();

}

void tree::computeFrobNorm( unsigned int l )
{
  for ( unsigned int i = 0; i < superNodeList_[l].size(); i++ )
    {
      for ( unsigned int j = 0; j < superNodeList_[l][i]->outEdges.size(); j++)
	{
	  frobNorms[l] += std::pow( superNodeList_[l][i]->outEdges[j]->matrix->norm(), 2);
	  totalSizes[l] +=  ( superNodeList_[l][i]->outEdges[j]->matrix->rows() * superNodeList_[l][i]->outEdges[j]->matrix->cols() );
	}
    }
  globFrobNorm = std::sqrt( std::pow( globFrobNorm, 2) + frobNorms[l] );
  globSize += totalSizes[l];
  frobNorms[l] = std::sqrt( frobNorms[l] );
  std::cout<<"frobNorm at level "<<l<<" = "<<frobNorms[l]<<" , total size = "<<totalSizes[l]<<"\n";
}

//! Top to bottom traverse
/*!
  Start from the top, solve the first set of equations directly (e.g., direct LU),
  Then back-propagate toword leaves, and solve all unknowns.
  This version uses the RHS provided in the input params.
*/
VectorXd& tree::solve()
{
  // solve
  return solve( *RHS_ );
}


// first set the RHS of all super/black nodes, then call solve()
VectorXd& tree::solve( VectorXd &RHS )
{

  // CALL SET RHS FOR THE GIVEN RHS
  setRHS( RHS );

  clock_t start, finish;
  start = clock();

  // std::cout<<"start bottom to top solve\n";
  //! bottom to top traverse to solve L z = b
  for ( unsigned int l = maxLevel(); l > 0; l-- )
    {
      solveL(l);
    }
  // std::cout<<"start top to bottom solve\n";
  //! top to bottom traverse to solve U x = z
  for ( unsigned int l = 1; l <= maxLevel(); l++ )
    {
      solveU(l);
    }
  finish = clock();

  return computeSolution();
}


//! Bottom to top traverse
/*!
  At each level, first create the superNodes (i.e., combine two redNodes), and then eliminate (and compress) all superNodes, as well as their black node parents.
*/
void tree::factorize()
{
  timeTuple4 eliminate_times;
  double merge_time;
  for ( unsigned int l = maxLevel(); l > 0; l-- )
    {
      merge_time = createSuperNodes(l);
      computeFrobNorm(l);
      eliminate_times = eliminate(l);
      std::cout<<"    Merge: "<<merge_time<<"   Cmp(SVD): "<<eliminate_times[0]<<"   Comp(etc.): "<<eliminate_times[1]<<"   S.C.(inv): "<<eliminate_times[2]<<"   S.C.(etc.): "<<eliminate_times[3]<<std::endl<<std::endl;
    }
}

VectorXd* tree::retVal()
{
  computeSolution();
  return VAR_;
}

void tree::normalize_cols( spMat* A)
{
  // each colmn
  for ( int i = 0; i < A -> outerSize(); i++ )
    {

      // find max of the column
      double max_entry = -1e30;
      for ( int j = A->outerIndexPtr()[i]; j < A->outerIndexPtr()[i+1]; j++ )
	{
	  if ( A -> valuePtr()[j] > max_entry )
	    {
	      max_entry = A -> valuePtr()[j];
	    }
	}
      max_entry = 1./max_entry;
      for ( int j = A->outerIndexPtr()[i]; j < A->outerIndexPtr()[i+1]; j++ )
	{
	  A -> valuePtr()[j] *= max_entry;
	}
    }

}
