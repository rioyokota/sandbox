#include "args.h"
#include "build_tree.h"
#include "dataset.h"
#include "ewald.h"
#include "kernel.h"
#include "timer.h"
#include "traverse.h"
#include "verify.h"
using namespace exafmm;

int main(int argc, char ** argv) {
  Args args(argc, argv);
  P = args.P;
  THETA = args.theta;
  NCRIT = args.ncrit;
  VERBOSE = args.verbose;
  IMAGES = args.images;
  CYCLE = 2 * M_PI;
  KSIZE = 11;
  ALPHA = KSIZE / CYCLE;
  SIGMA = .25 / M_PI;
  CUTOFF = CYCLE / 2;

  print("FMM Parameter");
  args.show();

  double totalFMM = 0;
  print("FMM Profiling");
  start("Initialize bodies");
  Bodies bodies = initBodies(args.numBodies, args.distribution);
  stop("Initialize bodies");
  start("Total FMM");
  start("Precalculation");
  initKernel();
  stop("Precalculation");
  start("Build tree");
  Cells cells = buildTree(bodies);
  stop("Build tree");
  start("P2M & M2M");
  upwardPass(cells);
  stop("P2M & M2M");
  start("M2L & P2P");
  horizontalPass(cells, cells);
  stop("M2L & P2P");
  start("L2L & L2P");
  downwardPass(cells);
  stop("L2L & L2P");
  totalFMM += stop("Total FMM");
  Bodies bodies2;
  if ((IMAGES == 0) | (bodies.size() < 1000)) {
    start("Direct N-Body");
    const int numTargets = 10;
    Bodies jbodies = bodies;
    sampleBodies(bodies, numTargets);
    bodies2 = bodies;
    initTarget(bodies);
    direct(bodies, jbodies);
    stop("Direct N-Body");
  } else {
    start("Dipole correction");
    vec3 dipole = 0;
    for (size_t b=0; b<bodies.size(); b++) dipole += bodies[b].X * bodies[b].q;
    real_t coef = 4 * M_PI / (3 * CYCLE * CYCLE * CYCLE);
    real_t dnorm = norm(dipole) / bodies.size();
    for (size_t b=0; b<bodies.size(); b++) {
      bodies[b].p -= coef * dnorm / bodies[b].q;
      bodies[b].F -= dipole * coef;
    }
    stop("Dipole correction");

    print("Ewald Profiling");
    start("Build tree");
    bodies2 = bodies;
    initTarget(bodies);
    Bodies jbodies = bodies;
    Cells  jcells = buildTree(jbodies);
    stop("Build tree");
    start("Wave part");
    wavePart(bodies, jbodies);
    stop("Wave part");
    start("Real part");
    realPart(cells, jcells);
    selfTerm(bodies);
    stop("Real part");
  }

  double pDif, pNrm;
  if (IMAGES == 0) {
    pDif = getDifScalar(bodies, bodies2);
    pNrm = getNrmScalar(bodies2);
  } else {
    double pSum = getSumScalar(bodies);
    double pSum2 = getSumScalar(bodies2);
    pDif = (pSum - pSum2) * (pSum - pSum2);
    pNrm = pSum * pSum;
  }
  double pRel = std::sqrt(pDif/pNrm);
  double FDif = getDifVector(bodies, bodies2);
  double FNrm = getNrmVector(bodies2);
  double FRel = std::sqrt(FDif/FNrm);
  print("FMM vs. direct");
  print("Rel. L2 Error (p)", pRel, false);
  print("Rel. L2 Error (F)", FRel, false);
  return 0;
}
