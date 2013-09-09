#include "Treecode.h"

int main(int argc, char * argv[])
{
  typedef Treecode Tree;

  const int nBody = 16777216;
  const int seed = 19810614;
  const float eps   = 0.05;
  const float theta = 0.75;
  const int ncrit = 64;
  const int nleaf = 64;
  Tree tree(eps, theta);

  fprintf(stdout,"--- FMM Parameters ---------------\n");
  fprintf(stdout,"numBodies            : %d\n",nBody);
  fprintf(stdout,"P                    : %d\n",3);
  fprintf(stdout,"theta                : %f\n",theta);
  fprintf(stdout,"ncrit                : %d\n",ncrit);
  fprintf(stdout,"nleaf                : %d\n",nleaf);
  const Plummer data(nBody, seed);
  tree.alloc(nBody);
  for (int i = 0; i < nBody; i++) {
    float4 bodyPos, bodyVel, bodyAcc;
    bodyPos.x    = data.pos[i].x;
    bodyPos.y    = data.pos[i].y;
    bodyPos.z    = data.pos[i].z;
    bodyPos.w    = data.mass[i];

    bodyVel.x    = data.vel[i].x;
    bodyVel.y    = data.vel[i].y;
    bodyVel.z    = data.vel[i].z;
    bodyVel.w    = data.mass[i];

    bodyAcc.x    = 0;
    bodyAcc.y    = 0;
    bodyAcc.z    = 0;
    bodyAcc.w    = 0;

    tree.h_bodyPos[i] = bodyPos;
    tree.h_bodyVel[i] = bodyVel;
    tree.h_bodyAcc[i] = bodyAcc;
    tree.h_bodyAcc2[i] = make_float4(0,0,0,0);
  }

  tree.body_h2d();

  fprintf(stdout,"--- FMM Profiling ----------------\n");
  double t0 = get_time();
  tree.buildTree(nleaf); // pass nLeaf, accepted 16, 24, 32, 48, 64
  tree.computeMultipoles();
  tree.groupTargets(5, ncrit); // pass nCrit
  const float4 interactions = tree.computeForces();
  double dt = get_time() - t0;
  float flops = (interactions.x*20 + interactions.z*64)*tree.get_nBody()/dt/1e12;
  fprintf(stdout,"--- Total runtime ----------------\n");
  fprintf(stdout,"Total FMM            : %.7f s (%.7f TFlops)\n",dt,flops);
  const int numTarget = 512; // Number of threads per block will be set to this value
  const int numBlock = 128;
  t0 = get_time();
  tree.computeDirect(numTarget,numBlock);
  dt = get_time() - t0;
  flops = 20.*numTarget*nBody/dt/1e12;
  fprintf(stdout,"Total Direct         : %.7f s (%.7f TFlops)\n",dt,flops);
  tree.body_d2h();

  for (int i=0; i<numTarget; i++) {
    float4 bodyAcc = tree.h_bodyAcc2[i];
    for (int j=1; j<numBlock; j++) {
      bodyAcc.x += tree.h_bodyAcc2[i+numTarget*j].x;
      bodyAcc.y += tree.h_bodyAcc2[i+numTarget*j].y;
      bodyAcc.z += tree.h_bodyAcc2[i+numTarget*j].z;
      bodyAcc.w += tree.h_bodyAcc2[i+numTarget*j].w;
    }
    tree.h_bodyAcc2[i] = bodyAcc;
  }

  double diffp = 0, diffa = 0;
  double normp = 0, norma = 0;
  for (int i=0; i<numTarget; i++) {
    diffp += (tree.h_bodyAcc[i].w - tree.h_bodyAcc2[i].w) * (tree.h_bodyAcc[i].w - tree.h_bodyAcc2[i].w);
    diffa += (tree.h_bodyAcc[i].x - tree.h_bodyAcc2[i].x) * (tree.h_bodyAcc[i].x - tree.h_bodyAcc2[i].x)
      + (tree.h_bodyAcc[i].y - tree.h_bodyAcc2[i].y) * (tree.h_bodyAcc[i].y - tree.h_bodyAcc2[i].y)
      + (tree.h_bodyAcc[i].z - tree.h_bodyAcc2[i].z) * (tree.h_bodyAcc[i].z - tree.h_bodyAcc2[i].z);
    normp += tree.h_bodyAcc2[i].w * tree.h_bodyAcc2[i].w;
    norma += tree.h_bodyAcc2[i].x * tree.h_bodyAcc2[i].x
      + tree.h_bodyAcc2[i].y * tree.h_bodyAcc2[i].y
      + tree.h_bodyAcc2[i].z * tree.h_bodyAcc2[i].z;
  }
  fprintf(stdout,"--- FMM vs. direct ---------------\n");
  fprintf(stdout,"Rel. L2 Error (pot)  : %.7e\n",sqrt(diffp/normp));
  fprintf(stdout,"Rel. L2 Error (acc)  : %.7e\n",sqrt(diffa/norma));
  fprintf(stdout,"--- Tree stats -------------------\n");
  fprintf(stdout,"Bodies               : %d\n",tree.get_nBody());
  fprintf(stdout,"Cells                : %d\n",tree.getNumSources());
  fprintf(stdout,"Tree depth           : %d\n",tree.getNumLevels());
  fprintf(stdout,"--- Traversal stats --------------\n");
  fprintf(stdout,"P2P mean list length : %g (max %g)\n", interactions.x, interactions.y);
  fprintf(stdout,"M2P mean list length : %g (max %g)\n", interactions.z, interactions.w);
  return 0;
}