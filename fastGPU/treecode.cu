#include "Treecode.h"

int main(int argc, char * argv[])
{
  typedef Treecode Tree;

  const int numBodies = 16777216;
  const int seed = 19810614;
  const float eps   = 0.05;
  const float THETA = 0.75;
  const int NCRIT = 64;
  const int NLEAF = 64;
  Tree tree(eps, THETA);

  fprintf(stdout,"--- FMM Parameters ---------------\n");
  fprintf(stdout,"numBodies            : %d\n",numBodies);
  fprintf(stdout,"P                    : %d\n",3);
  fprintf(stdout,"THETA                : %f\n",THETA);
  fprintf(stdout,"NCRIT                : %d\n",NCRIT);
  fprintf(stdout,"NLEAF                : %d\n",NLEAF);
  const Plummer data(numBodies, seed);

  host_mem<float4> h_bodyPos;
  h_bodyPos.alloc(numBodies);
  
  tree.alloc(numBodies);
  for (int i = 0; i < numBodies; i++) {
    float4 bodyPos;
    bodyPos.x    = data.pos[i].x;
    bodyPos.y    = data.pos[i].y;
    bodyPos.z    = data.pos[i].z;
    bodyPos.w    = data.mass[i];
    h_bodyPos[i] = bodyPos;
  }
  tree.d_bodyPos.h2d(h_bodyPos);
  tree.d_bodyAcc2.h2d(h_bodyPos);

  cuda_mem<float4> d_domain;
  cuda_mem<int2> d_levelRange;
  d_domain.alloc(1);
  d_levelRange.alloc(32);

  fprintf(stdout,"--- FMM Profiling ----------------\n");
  double t0 = get_time();
  tree.buildTree(d_domain, d_levelRange, NLEAF); // pass NLEAF, accepted 16, 24, 32, 48, 64
  tree.computeMultipoles();
  tree.groupTargets(d_domain, 5, NCRIT);
  const float4 interactions = tree.computeForces(d_levelRange);
  double dt = get_time() - t0;
  float flops = (interactions.x * 20 + interactions.z * 64) * tree.getNumBody() / dt / 1e12;
  fprintf(stdout,"--- Total runtime ----------------\n");
  fprintf(stdout,"Total FMM            : %.7f s (%.7f TFlops)\n",dt,flops);
  const int numTarget = 512; // Number of threads per block will be set to this value
  const int numBlock = 128;
  t0 = get_time();
  tree.computeDirect(numTarget,numBlock);
  dt = get_time() - t0;
  flops = 20.*numTarget*numBodies/dt/1e12;
  fprintf(stdout,"Total Direct         : %.7f s (%.7f TFlops)\n",dt,flops);
  host_mem<float4> h_bodyAcc, h_bodyAcc2;
  h_bodyAcc.alloc(numBodies);
  h_bodyAcc2.alloc(numBodies);
  tree.d_bodyAcc.d2h(h_bodyAcc);
  tree.d_bodyAcc2.d2h(h_bodyAcc2);

  for (int i=0; i<numTarget; i++) {
    float4 bodyAcc = h_bodyAcc2[i];
    for (int j=1; j<numBlock; j++) {
      bodyAcc.x += h_bodyAcc2[i+numTarget*j].x;
      bodyAcc.y += h_bodyAcc2[i+numTarget*j].y;
      bodyAcc.z += h_bodyAcc2[i+numTarget*j].z;
      bodyAcc.w += h_bodyAcc2[i+numTarget*j].w;
    }
    h_bodyAcc2[i] = bodyAcc;
  }

  double diffp = 0, diffa = 0;
  double normp = 0, norma = 0;
  for (int i=0; i<numTarget; i++) {
    diffp += (h_bodyAcc[i].w - h_bodyAcc2[i].w) * (h_bodyAcc[i].w - h_bodyAcc2[i].w);
    diffa += (h_bodyAcc[i].x - h_bodyAcc2[i].x) * (h_bodyAcc[i].x - h_bodyAcc2[i].x)
      + (h_bodyAcc[i].y - h_bodyAcc2[i].y) * (h_bodyAcc[i].y - h_bodyAcc2[i].y)
      + (h_bodyAcc[i].z - h_bodyAcc2[i].z) * (h_bodyAcc[i].z - h_bodyAcc2[i].z);
    normp += h_bodyAcc2[i].w * h_bodyAcc2[i].w;
    norma += h_bodyAcc2[i].x * h_bodyAcc2[i].x
      + h_bodyAcc2[i].y * h_bodyAcc2[i].y
      + h_bodyAcc2[i].z * h_bodyAcc2[i].z;
  }
  fprintf(stdout,"--- FMM vs. direct ---------------\n");
  fprintf(stdout,"Rel. L2 Error (pot)  : %.7e\n",sqrt(diffp/normp));
  fprintf(stdout,"Rel. L2 Error (acc)  : %.7e\n",sqrt(diffa/norma));
  fprintf(stdout,"--- Tree stats -------------------\n");
  fprintf(stdout,"Bodies               : %d\n",tree.getNumBody());
  fprintf(stdout,"Cells                : %d\n",tree.getNumSources());
  fprintf(stdout,"Tree depth           : %d\n",tree.getNumLevels());
  fprintf(stdout,"--- Traversal stats --------------\n");
  fprintf(stdout,"P2P mean list length : %g (max %g)\n", interactions.x, interactions.y);
  fprintf(stdout,"M2P mean list length : %g (max %g)\n", interactions.z, interactions.w);
  return 0;
}