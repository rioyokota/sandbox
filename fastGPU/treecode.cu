#include "Treecode.h"

int main(int argc, char * argv[])
{
  typedef Treecode Tree;

  const int nPtcl = 16777216;
  const int seed = 19810614;
  const float eps   = 0.05;
  const float theta = 0.75;
  const int ncrit = 64;
  const int nleaf = 64;
  Tree tree(eps, theta);

  fprintf(stdout,"--- FMM Parameters ---------------\n");
  fprintf(stdout,"numBodies            : %d\n",nPtcl);
  fprintf(stdout,"P                    : %d\n",3);
  fprintf(stdout,"theta                : %f\n",theta);
  fprintf(stdout,"ncrit                : %d\n",ncrit);
  fprintf(stdout,"nleaf                : %d\n",nleaf);
  const Plummer data(nPtcl, seed);
  tree.alloc(nPtcl);
  for (int i = 0; i < nPtcl; i++) {
    float4 ptclPos, ptclVel, ptclAcc;
    ptclPos.x    = data.pos[i].x;
    ptclPos.y    = data.pos[i].y;
    ptclPos.z    = data.pos[i].z;
    ptclPos.w    = data.mass[i];

    ptclVel.x    = data.vel[i].x;
    ptclVel.y    = data.vel[i].y;
    ptclVel.z    = data.vel[i].z;
    ptclVel.w    = data.mass[i];

    ptclAcc.x    = 0;
    ptclAcc.y    = 0;
    ptclAcc.z    = 0;
    ptclAcc.w    = 0;

    tree.h_ptclPos[i] = ptclPos;
    tree.h_ptclVel[i] = ptclVel;
    tree.h_ptclAcc[i] = ptclAcc;
    tree.h_ptclAcc2[i] = make_float4(0,0,0,0);
  }

  tree.ptcl_h2d();

  fprintf(stdout,"--- FMM Profiling ----------------\n");
  double t0 = get_time();
  tree.buildTree(nleaf); // pass nLeaf, accepted 16, 24, 32, 48, 64
  tree.computeMultipoles();
  tree.makeGroups(5, ncrit); // pass nCrit
  const float4 interactions = tree.computeForces();
  double dt = get_time() - t0;
#ifdef QUADRUPOLE
  const int FLOPS_QUAD = 64;
#else
  const int FLOPS_QUAD = 20;
#endif
  float flops = (interactions.x*20 + interactions.z*FLOPS_QUAD)*tree.get_nPtcl()/dt/1e12;
  fprintf(stdout,"--- Total runtime ----------------\n");
  fprintf(stdout,"Total FMM            : %.7f s (%.7f TFlops)\n",dt,flops);
  const int numTarget = 512; // Number of threads per block will be set to this value
  const int numBlock = 128;
  t0 = get_time();
  tree.computeDirect(numTarget,numBlock);
  dt = get_time() - t0;
  flops = 20.*numTarget*nPtcl/dt/1e12;
  fprintf(stdout,"Total Direct         : %.7f s (%.7f TFlops)\n",dt,flops);
  tree.ptcl_d2h();

  for (int i=0; i<numTarget; i++) {
    float4 ptclAcc = tree.h_ptclAcc2[i];
    for (int j=1; j<numBlock; j++) {
      ptclAcc.x += tree.h_ptclAcc2[i+numTarget*j].x;
      ptclAcc.y += tree.h_ptclAcc2[i+numTarget*j].y;
      ptclAcc.z += tree.h_ptclAcc2[i+numTarget*j].z;
      ptclAcc.w += tree.h_ptclAcc2[i+numTarget*j].w;
    }
    tree.h_ptclAcc2[i] = ptclAcc;
  }

  double diffp = 0, diffa = 0;
  double normp = 0, norma = 0;
  for (int i=0; i<numTarget; i++) {
    diffp += (tree.h_ptclAcc[i].w - tree.h_ptclAcc2[i].w) * (tree.h_ptclAcc[i].w - tree.h_ptclAcc2[i].w);
    diffa += (tree.h_ptclAcc[i].x - tree.h_ptclAcc2[i].x) * (tree.h_ptclAcc[i].x - tree.h_ptclAcc2[i].x)
      + (tree.h_ptclAcc[i].y - tree.h_ptclAcc2[i].y) * (tree.h_ptclAcc[i].y - tree.h_ptclAcc2[i].y)
      + (tree.h_ptclAcc[i].z - tree.h_ptclAcc2[i].z) * (tree.h_ptclAcc[i].z - tree.h_ptclAcc2[i].z);
    normp += tree.h_ptclAcc2[i].w * tree.h_ptclAcc2[i].w;
    norma += tree.h_ptclAcc2[i].x * tree.h_ptclAcc2[i].x
      + tree.h_ptclAcc2[i].y * tree.h_ptclAcc2[i].y
      + tree.h_ptclAcc2[i].z * tree.h_ptclAcc2[i].z;
  }
  fprintf(stdout,"--- FMM vs. direct ---------------\n");
  fprintf(stdout,"Rel. L2 Error (pot)  : %.7e\n",sqrt(diffp/normp));
  fprintf(stdout,"Rel. L2 Error (acc)  : %.7e\n",sqrt(diffa/norma));
  fprintf(stdout,"--- Tree stats -------------------\n");
  fprintf(stdout,"Bodies               : %d\n",tree.get_nPtcl());
  fprintf(stdout,"Cells                : %d\n",tree.get_nCells());
  fprintf(stdout,"Tree depth           : %d\n",tree.get_nLevels());
  fprintf(stdout,"--- Traversal stats --------------\n");
  fprintf(stdout,"P2P mean list length : %g (max %g)\n", interactions.x, interactions.y);
  fprintf(stdout,"M2P mean list length : %g (max %g)\n", interactions.z, interactions.w);
  return 0;
}