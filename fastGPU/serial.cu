#include "types.h"
#include "buildtree.h"
#include "dataset.h"
#include "grouptargets.h"
#include "traversal.h"
#include "upwardpass.h"

int main(int argc, char ** argv) {
  const int numBodies = (1 << 24) - 1;
  const float eps = 0.05;
  const float theta = 0.75;
  const int ncrit = 64;

  fprintf(stdout,"--- FMM Parameters ---------------\n");
  fprintf(stdout,"numBodies            : %d\n",numBodies);
  fprintf(stdout,"P                    : %d\n",3);
  fprintf(stdout,"theta                : %f\n",theta);
  fprintf(stdout,"ncrit                : %d\n",ncrit);
  const Dataset data(numBodies);

  cudaVec<float4> bodyPos(numBodies,true);
  cudaVec<float4> bodyPos2(numBodies);
  cudaVec<float4> bodyAcc(numBodies,true);
  cudaVec<float4> bodyAcc2(numBodies,true);
  for (int i=0; i<numBodies; i++) {
    bodyPos[i].x = data.pos[i].x;
    bodyPos[i].y = data.pos[i].y;
    bodyPos[i].z = data.pos[i].z;
    bodyPos[i].w = data.pos[i].w;
  }
  bodyPos.h2d();
  bodyAcc.h2d();

  fprintf(stdout,"--- FMM Profiling ----------------\n");
  double t0 = get_time();
  Build build;
  float4 domain;
  cudaVec<int2> levelRange(32,true);
  cudaVec<CellData> sourceCells(numBodies);
  int3 counts = build.tree<ncrit>(bodyPos, bodyPos2, domain, levelRange, sourceCells);
  int numLevels = counts.x;
  int numSources = counts.y;
  int numLeafs = counts.z;
  cudaVec<int2> targetRange(numBodies);
  cudaVec<float4> sourceCenter(numSources);
  cudaVec<float4> Multipole(3*numSources);
  Group group;
  int numTargets = group.targets(bodyPos, bodyPos2, domain, targetRange, 5);
  Pass pass;
  pass.upward(numLeafs, numLevels, theta, levelRange, bodyPos, sourceCells, sourceCenter, Multipole);
  Traversal traversal;
  const float4 interactions = traversal.approx(numTargets, eps,
					       bodyPos, bodyPos2, bodyAcc,
					       targetRange, sourceCells, sourceCenter,
					       Multipole, levelRange);
  double dt = get_time() - t0;
  float flops = (interactions.x * 20 + interactions.z * 64) * numBodies / dt / 1e12;
  fprintf(stdout,"--- Total runtime ----------------\n");
  fprintf(stdout,"Total FMM            : %.7f s (%.7f TFlops)\n",dt,flops);
  const int numTarget = min(512,numBodies); // Number of threads per block will be set to this value
  const int numBlock = min(128,(numBodies-1)/numTarget+1);
  t0 = get_time();
  traversal.direct(numTarget, numBlock, eps, bodyPos2, bodyAcc2);
  dt = get_time() - t0;
  flops = 35.*numTarget*numBodies/dt/1e12;
  fprintf(stdout,"Total Direct         : %.7f s (%.7f TFlops)\n",dt,flops);
  bodyAcc.d2h();
  bodyAcc2.d2h();

  for (int i=0; i<numTarget; i++) {
    float4 bodyAcc = bodyAcc2[i];
    for (int j=1; j<numBlock; j++) {
      bodyAcc.x += bodyAcc2[i+numTarget*j].x;
      bodyAcc.y += bodyAcc2[i+numTarget*j].y;
      bodyAcc.z += bodyAcc2[i+numTarget*j].z;
      bodyAcc.w += bodyAcc2[i+numTarget*j].w;
    }
    bodyAcc2[i] = bodyAcc;
  }

  double diffp = 0, diffa = 0;
  double normp = 0, norma = 0;
  for (int i=0; i<numTarget; i++) {
    diffp += (bodyAcc[i].w - bodyAcc2[i].w) * (bodyAcc[i].w - bodyAcc2[i].w);
    diffa += (bodyAcc[i].x - bodyAcc2[i].x) * (bodyAcc[i].x - bodyAcc2[i].x)
      + (bodyAcc[i].y - bodyAcc2[i].y) * (bodyAcc[i].y - bodyAcc2[i].y)
      + (bodyAcc[i].z - bodyAcc2[i].z) * (bodyAcc[i].z - bodyAcc2[i].z);
    normp += bodyAcc2[i].w * bodyAcc2[i].w;
    norma += bodyAcc2[i].x * bodyAcc2[i].x
      + bodyAcc2[i].y * bodyAcc2[i].y
      + bodyAcc2[i].z * bodyAcc2[i].z;
  }
  fprintf(stdout,"--- FMM vs. direct ---------------\n");
  fprintf(stdout,"Rel. L2 Error (pot)  : %.7e\n",sqrt(diffp/normp));
  fprintf(stdout,"Rel. L2 Error (acc)  : %.7e\n",sqrt(diffa/norma));
  fprintf(stdout,"--- Tree stats -------------------\n");
  fprintf(stdout,"Bodies               : %d\n",numBodies);
  fprintf(stdout,"Cells                : %d\n",numSources);
  fprintf(stdout,"Tree depth           : %d\n",numLevels);
  fprintf(stdout,"--- Traversal stats --------------\n");
  fprintf(stdout,"P2P mean list length : %g (max %g)\n", interactions.x, interactions.y);
  fprintf(stdout,"M2P mean list length : %g (max %g)\n", interactions.z, interactions.w);
  return 0;
}