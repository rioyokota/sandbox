#include "superNode.h"
#include "redNode.h"
#include "tree.h"
#include "edge.h"
#include "Eigen/Dense"
#include "Eigen/LU"
#include <iostream>
#include "params.h"
#include <cassert>
#include "rsvd.h"

// Constructor for superNode
superNode::superNode( blackNode* P, tree* T, int first, int last): node( (node*) P, T, first, last )
{
  // set the machine precision zero
  MACHINE_PRECISION = 1e-14;

  // Set its parent
  parent_ = P;
  
  // Set its level
  level_ = P->parent()->level()+1;
  
  // When a superNode born, it should be added to the superNode list of the tree
  T->addSuperNode(level_,this);

  // We set m, and n based on number of columns/rows corresponding to this redNode.
  // However, for non-leaf redNodes, m and n will be changed after low-rank approximation
  m( last - first + 1 );
  n( last - first + 1 );

  // Initially not eliminated
  eliminated_ = false;

  // Get values for compression accuracy
  epsilon_ = T->Parameters()->epsilon();

  // Get values for compression low-rank method
  lowRankMethod_ = T->Parameters()->lowRankMethod();

  // Get values for compression cut-off method
  cutOffMethod_ = T->Parameters()->cutOffMethod();

  // GET A PRIORY RANK FOR COMPRESSOIN
  aPrioriRank_ = T->Parameters()->aPrioriRank();

  // Get rank cap factor for compression
  rankCapFactor_ = T->Parameters()->rankCapFactor();

  // Get deploy factor for compression
  deployFactor_ = T->Parameters()->deployFactor();
  
  // Pointer to the frobNorm list:
  frobNorms_ = &( T->frobNorms[0] );

  // Pointer to the globFrobNorm of the tree;
  globFrobNorm_ = &(T -> globFrobNorm);

  // Pointer to the globSize of the tree;
  globSize_ = &(T -> globSize);

  // Pointer to the totalSizes list:
  totalSizes_ = &( T->totalSizes[0] );

  // Tree depth
  treeDepth_ = T->Parameters()->treeLevelThreshold();
 
}

// Destructor
superNode::~superNode()
{
  // Delete interaction edges
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      delete outEdges[i];
    }
  
  // Delete the inverse of the pivot matrix
  delete invPivot;
}

redNode* superNode::redParent( int l )
{
  // cross-recursive !

  if ( l == 0 )
    {
      return parent_->parent();
    }
  else
    {
      return redParent()->redParent( l );
    }
}

// Compress all well-separated interactions
void superNode::compress()
{

  // Check if superNode has any variable
  if ( m() == 0 )
    {
      parent() -> m( 0 );
      parent() -> n( 0 );
      redParent() -> m( 0 );
      redParent() -> n( 0 );
      return;
    }
  
  // First erase all previously compressed edges from in/out edge lists
  eraseCompressedEdges();
  
  // Concatenate U matrices:
  // calculate the total number of columns
  int nCols = 0;
  // Go over all incoming edges, and compress those are well-separated
  for ( unsigned int i = 0; i < inEdges.size(); i++ )
    {
      // Check if the edge is well-separated
      if ( inEdges[i] -> isWellSeparated() )
	{
	  nCols += inEdges[i] -> matrix ->cols(); 
	}
    } 
  densMat bigU( m(), nCols );
  int nColsFilled = 0;
  for ( unsigned int i = 0; i < inEdges.size(); i++ )
    {
      // Check if the edge is well-separated
      if ( inEdges[i] -> isWellSeparated() )
	{
	  bigU.block( 0, nColsFilled, m(), inEdges[i]->matrix->cols() ) << *(inEdges[i]->matrix);
	  nColsFilled += inEdges[i]->matrix->cols();
	}
    }

  // concatenate outGoing edges:
    // calculate the total number of rows
  int nRows = 0;
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      // Check if the edge is well-separated
      if ( outEdges[i] -> isWellSeparated() )
	{
	  nRows += outEdges[i]->matrix->rows();
	}
    }
  densMat bigV( nRows, n() );
  int nRowsFilled = 0;
    for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      // Check if the edge is well-separated
      if ( outEdges[i] -> isWellSeparated() )
	{
	  bigV.block( nRowsFilled, 0, outEdges[i]->matrix->rows(), n() ) << *( outEdges[i]->matrix );
	  nRowsFilled += outEdges[i]->matrix->rows();
	}
    }

    int kU = 0;
    int kV = 0;
    
    if ( (nCols == 0) && (nRows == 0) )
      {
	parent() -> m( 0 );
	parent() -> n( 0 );
	redParent() -> m( 0 );
	redParent() -> n( 0 );
	return;
      }
    
    if ( (nCols == 0) || (nRows == 0) )
      {
	std::cout<<"TROUBLE!\n";
	parent() -> m( 0 );
	parent() -> n( 0 );
	redParent() -> m( 0 );
	redParent() -> n( 0 );
	return;
      }

    // Now we should reCommpress bigU and bigV with the same rank (and smaller than m() and n())
    // reCompress bigU = Unew * R and bigV = L * Vnew
    Eigen::JacobiSVD<densMat> svdU = Eigen::JacobiSVD<densMat>( bigU, Eigen::ComputeThinU | Eigen::ComputeThinV );
    kU = cutOff( svdU, cutOffMethod_, epsilon_, aPrioriRank_, rankCapFactor_ );
  
    Eigen::JacobiSVD<densMat> svdV = Eigen::JacobiSVD<densMat>( bigV, Eigen::ComputeThinU | Eigen::ComputeThinV );
    kV = cutOff( svdV, cutOffMethod_, epsilon_, aPrioriRank_, rankCapFactor_ );
  
    // take the maximum: This is going to be the size (Rank) of both interpolation and anterpolation operators associated to this superNode
    int k = std::max(kU , kV );
  
    // The variables m, and n for the parent and redParent now should be k
    parent() -> m( k );
    parent() -> n( k );
    redParent() -> m( k );
    redParent() -> n( k ); 
    
    if ( k == 0 )
      {
	for ( unsigned int i = 0; i < inEdges.size(); i++ )
	  {
	    // Check if the edge is well-separated
	    if ( inEdges[i] -> isWellSeparated() )
	      {
		inEdges[i]->compress();
	      }
	  }
	for ( unsigned int i = 0; i < outEdges.size(); i++ )
	  {
	    // Check if the edge is well-separated
	    if ( outEdges[i] -> isWellSeparated() )
	      {
		outEdges[i]->compress();
	      }
	  }
	eraseCompressedEdges();
	return;
      }
    
    // Now that we have the exact size of parents, we can set RHS and VAR
    parent() -> RHS( new VectorXd( parent() -> m() ) );
    //*( parent() -> RHS() ) = VectorXd::Zero( parent() -> m() );
    parent() -> RHS() -> setZero();
    parent() -> VAR( new VectorXd( parent() -> n() ) );
    
    redParent() -> RHS( new VectorXd( redParent() -> m() ) );
    redParent() -> RHS() -> setZero();
    redParent() -> VAR( new VectorXd( redParent() -> n() ) );

    // Create incoming edge from the balck parent
    parent() -> outEdges.push_back( new edge( parent(), this ) );
    inEdges.push_back( parent() -> outEdges.back() );
    // Assign memory for the matrix Unew (from parent to this)
    inEdges.back() -> matrix = new densMat( m(), k);

    // Create outgoing edge to the balck parent
    outEdges.push_back( new edge( this, parent() ) );
    parent() -> inEdges.push_back( outEdges.back() );
    
    // Assign memory for the matrix Vnew (from this to parent)
    outEdges.back() -> matrix = new densMat( k, n() );


    // See if we need padding:
    int p1 = kV - nCols;
    int p2 = kU - nRows;

    if ( p1 > 0 )
      {
	std::cout<<"Col padding is required.";
	Tree()->padding = true;
	inEdges.back() -> matrix -> block( 0, 0, m(), nCols ) = svdU.matrixU();
	extendOrthoCols( *(inEdges.back() -> matrix) , p1 ) ;
	std::cout<<" "<<p1<<" cols added to the bigU matrix!\n";
      }
    else
      {
	// Put the the reCompressed matrix (Unew) on the edge from parent
	*( inEdges.back() -> matrix ) = svdU.matrixU().leftCols(k);
      }
    if ( p2 > 0 )
      {
	std::cout<<"Row padding is required.";
	outEdges.back() -> matrix -> block( 0, 0, nRows, n() ) = svdV.matrixV().transpose();
	extendOrthoRows( *(outEdges.back() -> matrix), p2 ) ;
	std::cout<<" "<<p2<<" rows added to the bigV matrix!\n";
      }
    else
      {
	// Put the the reCompressed matrix (Vnew) on the edge to parent
	*( outEdges.back() -> matrix ) = ( svdV.matrixV().leftCols(k) ).transpose();
      }
    
    int firstRow = 0;
    for ( unsigned int i = 0; i < inEdges.size(); i++ )
      {
	// Check if the edge is well-separated
	if ( inEdges[i] -> isWellSeparated() )
	  {    
	    // Say the matrix on the edge is K [ kold * nCols ]
	    // Update K = R [k * kold] * K, which will be [k * nCols]
	    int nColsi = inEdges[i] -> matrix -> cols();
	    redParent() -> inEdges.push_back( new edge( inEdges[i] -> source(), redParent() ) );
	    inEdges[i] -> source() -> outEdges.push_back( redParent() -> inEdges.back() );
	    redParent() -> inEdges.back() -> matrix = new densMat( k, nColsi );
	
	    if ( p1 <= 0 ) // i.e., no padding was required
	      {
		*(redParent()->inEdges.back()->matrix) = ( svdU.singularValues().head(k).asDiagonal() * ( svdU.matrixV().block( firstRow, 0, nColsi, k ) ).transpose() );
	      }
	    else // i.e. column padding has happend
	      {
		redParent()->inEdges.back()->matrix->block(0,0,nCols, nColsi) = ( svdU.singularValues().asDiagonal() * ( svdU.matrixV().block( firstRow, 0, nColsi, nCols ) ).transpose() );
		redParent()->inEdges.back()->matrix->block(nCols,0,p1,nColsi) = densMat::Zero( p1, nColsi );
	      }	
	    inEdges[i] -> compress();
	    firstRow += nColsi;
	  }
      }
  
    // Repeat all in the above for outEdges

    firstRow = 0;
    for ( unsigned int i = 0; i < outEdges.size(); i++ )
      {
	// Check if the edge is well-separated
	if ( outEdges[i] -> isWellSeparated() )
	  {
	    
	    // Say the matrix on the edge is K [ nRows * k ]
	    // Update K = K * L [kold * k] , which will be [nRows * k]
	    int nRowsi = outEdges[i] -> matrix -> rows();
	    redParent() -> outEdges.push_back( new edge( redParent(), outEdges[i] -> destination() ) );
	    outEdges[i] -> destination() -> inEdges.push_back( redParent() -> outEdges.back() );
	    redParent() -> outEdges.back() -> matrix  =  new densMat( nRowsi, k );
	    
	    if ( p2 <= 0 ) // i.e., no row padding was required
	      {
		*(redParent()->outEdges.back()->matrix) = ( svdV.matrixU().block( firstRow, 0, nRowsi, k ) * svdV.singularValues().head(k).asDiagonal() );
	      }
	    else // i.e., row padding has happened
	      {
		redParent()->outEdges.back()->matrix->block(0,0,nRowsi,nRows) = ( svdV.matrixU().block( firstRow, 0, nRowsi, nRows ) * svdV.singularValues().asDiagonal() );
		redParent()->outEdges.back()->matrix->block(0,nRows,nRowsi,p2) = densMat::Zero( nRowsi, p2 );
	      }
	    outEdges[i] -> compress();
	    firstRow += nRowsi;
	  }
      } 
    
    eraseCompressedEdges();
}

// Calculate k, the number of important singular values (used for compression)
int superNode::cutOff( Eigen::JacobiSVD<densMat> &svd, int method, double epsilon, int apRank, double rankCapFactor )
{
  int return_value = 0;
  // number of singular values
  int p = svd.singularValues().size();
  
  if ( p == 0 )
    {
      return 0;
    }
  
  if ( svd.singularValues()(0)< MACHINE_PRECISION )
    {
      return 0;
    }

  // method0 : cutofff based on the absolute value
  if ( method == 0 )
    {
     
      int k_return = p;
      for (int i = 0; i < p; i++ )
	{
	  // Find the first singular value that is smaller than epsilon 
	  if ( ( svd.singularValues() )(i) < epsilon )
	    {
	      k_return = i;
	      break;
	    }
	}
      
      return_value = k_return;
    }
  // method1: cutoff based on relative values
  else if ( method == 1 )
    {
      double sigma1 = svd.singularValues()(0);
      
      int k_return = p;
      for (int i = 0; i < p; i++ )
	{
	  // Find the first singular value that is smaller than epsilon 
	  if ( svd.singularValues()(i) < epsilon * sigma1 )
	    {
	      k_return = i;
	      break;
	    }
	}
      
      return_value = k_return;
    }  
  else if ( method == 2 )
    {      
      int k_return = p;
      for (int i = 0; i < p; i++ )
	{
	  // Find the first singular value that is smaller than epsilon 
	  if ( ( svd.singularValues() )(i) < epsilon * frobNorms_[ level_ ]  )
	    {
	      k_return = i;
	      break;
	    }
	}
      
      return_value = k_return;
    }
  else if ( method == 3 )
    {
      
      int k_return = p;
      
      for (int i = 0; i < p; i++ )
	{
	  // Find the first singular value that is smaller than epsilon 
	  if ( ( svd.singularValues() )(i) < epsilon * (*globFrobNorm_) )
	    {
	      k_return = i;
	      break;
	    }
	}
      
      return_value = k_return;
    }

  else if (method == 4 )
    {
      
      int k_return = p;
      double frobNorm2 = 0;
      for (int i = p-1; i >= 0; i-- )
	{
	  frobNorm2 += std::pow( svd.singularValues()(i), 2 );
	  // Find the first place that the frobNorm is greater than epsilon
	  if ( std::sqrt(frobNorm2) > frobNorms_[ level_ ] * epsilon )
	    {
	      k_return = i;
	      break;
	    }
	}
      if ( std::sqrt(frobNorm2) <= frobNorms_[ level_ ] * epsilon )
	{
	  return 0;
	}
      else
	{
	  return_value = k_return + 1;
	}
    }
  else if ( method == 5 )
    {
      
      int k_return = p;
      double frobNorm = 0;
      double blockSize = ( svd.matrixU().rows() * svd.matrixV().rows() );
      double epsilon2 = blockSize * std::pow(frobNorms_[ level_ ],2) * epsilon * epsilon / totalSizes_[ level_ ];
      //double epsilon2 = std::pow(frobNorms_[ level_ ],2) * epsilon;
      for (int i = p-1; i >= 0; i-- )
	{
	  frobNorm += std::pow( svd.singularValues()(i), 2 );
	  // Find the first place that the frobNorm is greater than epsilon
	  if ( frobNorm > epsilon2 )
	    {
	      k_return = i;
	      break;
	    }
	}
      if ( frobNorm <= epsilon2 )
	{
	  return 0;
	}
      else
	{
	  return_value = k_return + 1;
	}
    }
    else if (method == 6 )
    {
      
      int k_return = p;
      double frobNorm2 = 0;
      for (int i = p-1; i >= 0; i-- )
	{
	  frobNorm2 += std::pow( svd.singularValues()(i), 2 );
	  // Find the first place that the frobNorm is greater than epsilon
	  if ( std::sqrt(frobNorm2) > (*globFrobNorm_) * epsilon )
	    {
	      k_return = i;
	      break;
	    }
	}
      if ( std::sqrt(frobNorm2) <= (*globFrobNorm_) * epsilon )
	{
	  return 0;
	}
      else
	{
	  return_value = k_return + 1;
	}
    }
    else if ( method == 7 )
      {
	
	int k_return = p;
	double frobNorm = 0;
	double blockSize = ( svd.matrixU().rows() * svd.matrixV().rows() );
	double epsilon2 = blockSize * std::pow((*globFrobNorm_),2) * epsilon * epsilon / (*globSize_);
	//double epsilon2 = std::pow(frobNorms_[ level_ ],2) * epsilon;
	for (int i = p-1; i >= 0; i-- )
	  {
	    frobNorm += std::pow( svd.singularValues()(i), 2 );
	    // Find the first place that the frobNorm is greater than epsilon
	    if ( frobNorm > epsilon2 )
	      {
		k_return = i;
		break;
	      }
	  }
	if ( frobNorm <= epsilon2 )
	  {
	    return 0;
	  }
	else
	  {
	    return_value = k_return + 1;
	  }
      }
    else if ( method == 8 )
      {
	return_value = std::min( p, apRank );
      }
  if ( rankCapFactor > MACHINE_PRECISION )
    {
      int rankCap = (int) ( apRank * std::pow( rankCapFactor, treeDepth_-level_) );
      return std::min( return_value, rankCap );
    }
  else
    {
      return return_value;
    }
}

bool superNode::criterionCheck( int meth, double eps, RedSVD::RedSVD<densMat> &RSVD, densMat &A, int r )
{
  if ( r == 0 || (meth == 8) )
    {
      return true;
    }
  
  else if ( meth == 0 )
    {
      return ( RSVD.singularValues()(r-1) <= eps );
    }
  else if ( meth == 1 )
    {
      if ( RSVD.singularValues()(0)< MACHINE_PRECISION )
	{
	  return true;
	}
      else if ( r == 1 )
	{
	  return false;
	}
      else
	{
	  return ( RSVD.singularValues()(r-1) / RSVD.singularValues()(0) <= eps );
	}
    }
  else if ( meth == 2 )
    {
      return ( RSVD.singularValues()(r-1) <= eps * frobNorms_[ level_ ] );
    }
  else if ( meth == 3 )
    {
      return ( RSVD.singularValues()(r-1) <= eps * (*globFrobNorm_) );
    }
  else
    {
      densMat residu = A - RSVD.matrixU().leftCols(r) * RSVD.singularValues().head(r).asDiagonal() * RSVD.matrixV().leftCols(r).transpose();
      double normResidu = residu.norm();
      double blockSize = ( RSVD.matrixU().rows() * RSVD.matrixV().rows() );
      if ( meth == 4 )
	{
	  return ( normResidu <= eps * frobNorms_[ level_ ] );
	}
      else if ( meth == 5 )
	{
	  double epsilon2 = blockSize * std::pow(frobNorms_[ level_ ],2) * eps*eps / totalSizes_[ level_ ];
	  return ( normResidu*normResidu <= epsilon2 );
	}
      else if ( meth == 6 )
	{
	  return ( normResidu <= eps * (*globFrobNorm_) );
	}
      else if ( meth == 7 )
	{
	  double epsilon2 = blockSize * std::pow((*globFrobNorm_),2) * eps*eps / (*globSize_);
	  return ( normResidu*normResidu <= epsilon2 );
	}
    }
  return false;
}


// Go through all edges and 
void superNode::schurComp()
{

  // Step 0: Check if this node has any variable
  if ( n() == 0 )
    {
      eliminate();
      parent() -> leftChild() -> eliminate();
      parent() -> rightChild() -> eliminate();
      parent()->eliminate();
      return;
    }
  
  // First we detect the selfEdge (i.e., the edge from supernode to itself)
  edge* selfEdge = NULL;
  for ( unsigned int i = 0; i < outEdges.size(); i++ )
    {
      if ( outEdges[i] -> destination() == this )
	{
	  selfEdge = outEdges[i];
	  break;
	}
    }

  if ( selfEdge == NULL )
    {
      std::cout<<" ZERO PIVOT !"<<std::endl;
      std::exit(0);
    }

  // Now compute the inverse of the selfEdge matrix (i.e., inverse of the pivot)
  invPivot = new densMat( m(), n() );
  *invPivot = selfEdge -> matrix -> inverse();
  if ( invPivot->norm() > 1e9 )
    {
      // std::cout<<"(superNode) normInvPivot = "<< invPivot->norm()<<" \n";
      Tree()->largePivot = true;
    }
  if ( selfEdge->matrix->norm() > 1e9 )
    {
      // std::cout<<"(superNode) normPivot = "<< selfEdge->matrix->norm() <<" \n";
    }

  edge* X;
  edge* Y;
  // Loop over all incoming edges ( exclude the selfEdge ): X
    
  for ( unsigned int i = 0; i < inEdges.size(); i++ )
    {

      X = inEdges[i];
      if ( ( X -> source() != this ) && !( X -> isEliminated() ) )
	{
	  
	  // At this point we can compute invPivot * X
	  densMat factor = (*invPivot) * (*(X->matrix));	  
	    
	  // Loop over all outgoing edges ( exclude the selfEdge ):Y
	  for ( unsigned int j = 0; j < outEdges.size(); j++ )
	    {
	      
	      Y = outEdges[j];
	      if ( ( Y -> destination() != this ) && !( Y -> isEliminated() ) )
		{
		  
		  // Here we have ei -> selfEdge -> ej
		  // After elimination we have: i->j  with matrix = -X * invPivot * Y
		  // We first check if the edge from i to j already exist or not.
		  // if exist we just add the fill-in, otherwise create a new edge
		  edge* fillinEdge = NULL;
		  for ( unsigned int k = 0; k < X->source()->outEdges.size(); k++ )
		    {
		      if ( X->source()->outEdges[k]->destination() == Y->destination() )
			{
			  fillinEdge = X->source()->outEdges[k];
			  break;
			}
		    }
		  if ( fillinEdge == NULL )
		    {
		      X->source()->outEdges.push_back( new edge( X->source(), Y->destination() ) );
		      Y->destination()->inEdges.push_back( X->source()->outEdges.back() );
		      X->source()->outEdges.back()->matrix = new densMat( Y->matrix->rows(), X->matrix->cols() );
		      *( X->source()->outEdges.back()->matrix ) = - (*(Y->matrix)) * factor;
		    }
		  else
		    {
		      *(fillinEdge -> matrix) -= (*(Y->matrix)) * factor;
		    }		  
		}
	    }
	}
    }

  // Now, mark the node as eliminated
  eliminate();

  // eliminate left and right redChildren

  parent() -> leftChild() -> eliminate();
  parent() -> rightChild() -> eliminate();

  // After eliminating a superNode, we can immediately eliminate its black-parent
  parent()->schurComp();
}

// Split the vector VAR to the left and right redNodes
void superNode::splitVAR()
{

  if ( parent() -> leftChild() -> n() > 0 ) // i.e., if the left redNode has any unknowns
    {
      *( parent() -> leftChild() -> VAR() ) = VAR() -> segment( 0, parent() -> leftChild() -> n() );
    }

  if ( parent() -> rightChild() -> n() > 0 ) // i.e., if the right redNode has any unknowns
    {
      *( parent() -> rightChild() -> VAR() ) = VAR() -> segment( parent() -> leftChild() -> n(), parent() -> rightChild() -> n() );
    }

}


// Add a block of random columns to the right side of a matrix
void superNode::addRandomCols( densMat &A, int p ) 
{
  // Number of rows and cols:
  int m = A.rows();
  int n = A.cols();
  
  // Resize A: resizing is not conservative in Eigen!
  densMat B = A;
  A.resize( Eigen::NoChange, n + p );

  A.block( 0, 0, m, n ) = B;
  A.block( 0, n, m, p ) = densMat::Random( m, p );
  
}

// Add a block of random rows at the bottom of a matrix
void superNode::addRandomRows( densMat &A, int p ) 
{
  // Number of rows and cols:
  int m = A.rows();
  int n = A.cols();
  
  // Resize A
  densMat B = A;
  A.resize( m + p, Eigen::NoChange );

  A.block( 0, 0, m, n ) = B;
  A.block( m, 0, p, n ) = densMat::Random( p, n );
  
}

// Add a block of random columns to the right side of a matrix
void superNode::extendOrthoCols( densMat &A, int p ) 
{
  // Number of rows and cols:
  int m = A.rows();
  int n = A.cols() - p;
  
  /*
  // Resize A: resizing is not conservative in Eigen!
  densMat B = A;
  A.resize( Eigen::NoChange, n + p );
  
  A.block( 0, 0, m, n ) = B;
  */
  
  // Go through all columns and subtract from the new random col
  VectorXd newVec( m );
  for ( int i = 0; i < p; i++ )
    {
      newVec = VectorXd::Random( m );
      for ( int j = 0; j < n + i; j ++ )
	{
	  // Note: columns of A are supposed to have unit norm
	  double innerProd = newVec.dot( A.col(j) );
	  // subtract
	  newVec -= innerProd * A.col(j);
	}
      // normalize the new vector and add it to the matrix
      newVec /= newVec.norm();
      A.col( n + i ) = newVec;
    }

  // check if columns are orthogonal
  for ( int i = 0; i < n + p; i++ ) 
    {
      for ( int j = i+1; j < n + p; j++ ) 
	{
	  double innerProd = (A.col(i)).dot(A.col(j));
	  if ( innerProd / double(m)  > MACHINE_PRECISION )
	    {
	      std::cout<<" m = "<<m<<" n = "<<n<<" p = "<<p<<std::endl;
	      std::cout<<" i = "<<i<<" j = "<<j<<std::endl;
	      std::cout<<" inner product is = " << innerProd << std::endl;
	      std::exit(1);
	    }
	}
    }
}

// Add a block of random columns to the right side of a matrix
void superNode::extendOrthoRows( densMat &A, int p ) 
{
  // Number of rows and cols:
  int m = A.rows() - p;
  int n = A.cols();
  
  /*
  // Resize A: resizing is not conservative in Eigen!
  densMat B = A;
  A.resize( m + p, Eigen::NoChange );
  
  A.block( 0, 0, m, n ) = B;
  */

  // Go through all rows and subtract from the new random row
  VectorXd newVec( n );
  for ( int i = 0; i < p; i++ )
    {
      newVec = VectorXd::Random( n );
      for ( int j = 0; j < m + i; j ++ )
	{
	  // Note: columns of A are supposed to have unit norm
	  double innerProd = newVec.dot( A.row(j) );
	  // subtract
	  newVec -= innerProd * A.row(j).transpose();
	}
      // normalize the new vector and add it to the matrix
      newVec /= newVec.norm();
      A.row( m+i ) = newVec.transpose();
    }

    // check if columns are orthogonal
  for ( int i = 0; i < m + p; i++ ) 
    {
      for ( int j = i+1; j < m + p; j++ ) 
	{
	  double innerProd = (A.row(i)).dot(A.row(j));
	  if ( innerProd / double(n)  > MACHINE_PRECISION )
	    {
	      std::cout<<" m = "<<m<<" n = "<<n<<" p = "<<p<<std::endl;
	      std::cout<<" i = "<<i<<" j = "<<j<<std::endl;
	      std::cout<<" inner product is = " << innerProd << std::endl;
	      std::exit(1);
	    }
	}
    }

}

