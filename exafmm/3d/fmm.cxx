#include "args.h"
#include "bound_box.h"
#include "build_tree.h"
#include "dataset.h"
#include "ewald.h"
#include "kernel.h"
#include "logger.h"
#include "namespace.h"
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"
using namespace EXAFMM_NAMESPACE;

int main(int argc, char ** argv) {
  const int numBodies = 1000;
  const int ksize = 11;
  const vec3 cycle = 2 * M_PI;
  const real_t alpha = ksize / max(cycle);
  const real_t sigma = .25 / M_PI;
  const real_t cutoff = max(cycle) / 2;
  const real_t eps2 = 0.0;
  const complex_t wavek = complex_t(10.,1.) / real_t(2 * M_PI);
  Args args(argc, argv);
  args.numBodies = numBodies;
  args.images = 4;
  Bodies bodies, bodies2, jbodies, gbodies, buffer;
  BoundBox boundBox;
  Bounds bounds;
  BuildTree localTree(args.ncrit);
  Cells cells, jcells;
  Dataset data;
  Ewald ewald(ksize, alpha, sigma, cutoff, cycle);
  Kernel kernel(args.P, eps2, wavek);
  Traversal traversal(kernel, args.theta, args.nspawn, args.images, args.path);
  UpDownPass upDownPass(kernel);
  Verify verify(args.path);
  num_threads(args.threads);

  verify.verbose = args.verbose;
  logger::verbose = args.verbose;
  logger::path = args.path;
  logger::printTitle("Ewald Parameters");
  args.print(logger::stringLength);
  ewald.print(logger::stringLength);
  bodies = data.initBodies(args.numBodies, args.distribution);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->X *= cycle / (2 * M_PI);
  }
  buffer.reserve(bodies.size());
  logger::printTitle("FMM Profiling");
  logger::startTimer("Total FMM");
  logger::startPAPI();
  data.initTarget(bodies);
  bounds = boundBox.getBounds(bodies);
  cells = localTree.buildTree(bodies, buffer, bounds);
  upDownPass.upwardPass(cells);
  traversal.initListCount(cells);
  traversal.initWeight(cells);
  traversal.traverse(cells, cells, cycle, args.dual);
  upDownPass.downwardPass(cells);

  buffer = bodies;
  vec3 dipole = upDownPass.getDipole(bodies,0);
  upDownPass.dipoleCorrection(bodies, dipole, numBodies, cycle);
  bodies2 = bodies;
  data.initTarget(bodies);
  logger::printTitle("Ewald Profiling");
  logger::startTimer("Total Ewald");
  jbodies = bodies;
  bounds = boundBox.getBounds(jbodies);
  jcells = localTree.buildTree(jbodies, buffer, bounds);
  ewald.wavePart(bodies, jbodies);
  ewald.realPart(cells, jcells);
  ewald.selfTerm(bodies);
  logger::printTitle("Total runtime");
  logger::printTime("Total FMM");
  logger::stopTimer("Total Ewald");
  double potSum = verify.getSumScalar(bodies);
  double potSum2 = verify.getSumScalar(bodies2);
  double accDif = verify.getDifVector(bodies, bodies2);
  double accNrm = verify.getNrmVector(bodies);
  double potDif = (potSum - potSum2) * (potSum - potSum2);
  double potNrm = potSum * potSum;
  double potRel = std::sqrt(potDif/potNrm);
  double accRel = std::sqrt(accDif/accNrm);
  logger::printTitle("FMM vs. Ewald");
  verify.print("Rel. L2 Error (pot)",potRel);
  verify.print("Rel. L2 Error (acc)",accRel);
  localTree.printTreeData(cells);
  traversal.printTraversalData();
  return 0;
}
