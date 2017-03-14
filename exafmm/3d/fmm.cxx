#include "args.h"
#include "bound_box.h"
#include "build_tree.h"
#include "dataset.h"
#include "kernel.h"
#include "logger.h"
#include "namespace.h"
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"
using namespace EXAFMM_NAMESPACE;

int main(int argc, char ** argv) {
  const vec3 cycle = 2 * M_PI;
  const real_t eps2 = 0.0;
  const complex_t wavek = complex_t(10.,1.) / real_t(2 * M_PI);
  Args args(argc, argv);
  Bodies bodies, bodies2, jbodies, buffer;
  BoundBox boundBox;
  Bounds bounds;
  BuildTree buildTree(args.ncrit);
  Cells cells, jcells;
  Dataset data;
  Kernel kernel(args.P, eps2, wavek);
  Traversal traversal(kernel, args.theta, args.nspawn, args.images, args.path);
  UpDownPass upDownPass(kernel);
  Verify verify(args.path);
  num_threads(args.threads);

  verify.verbose = args.verbose;
  logger::verbose = args.verbose;
  logger::path = args.path;
  logger::printTitle("FMM Parameters");
  args.print(logger::stringLength);
  bodies = data.initBodies(args.numBodies, args.distribution, 0);
  buffer.reserve(bodies.size());
  if (args.IneJ) {
    for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
      B->X[0] += M_PI;
      B->X[0] *= 0.5;
    }
    jbodies = data.initBodies(args.numBodies, args.distribution, 1);
    for (B_iter B=jbodies.begin(); B!=jbodies.end(); B++) {
      B->X[0] -= M_PI;
      B->X[0] *= 0.5;
    }
  }
  logger::printTitle("FMM Profiling");
  logger::startTimer("Total FMM");
  bounds = boundBox.getBounds(bodies);
  if (args.IneJ) {
    bounds = boundBox.getBounds(jbodies, bounds);
  }
  cells = buildTree.buildTree(bodies, buffer, bounds);
  upDownPass.upwardPass(cells);
  traversal.initListCount(cells);
  traversal.initWeight(cells);
  if (args.IneJ) {
    jcells = buildTree.buildTree(jbodies, buffer, bounds);
    upDownPass.upwardPass(jcells);
    traversal.traverse(cells, jcells, cycle, args.dual);
  } else {
    traversal.traverse(cells, cells, cycle, args.dual);
    jbodies = bodies;
  }
  upDownPass.downwardPass(cells);
  logger::printTitle("Total runtime");

  const int numTargets = 100;
  buffer = bodies;
  data.sampleBodies(bodies, numTargets);
  bodies2 = bodies;
  data.initTarget(bodies);
  logger::startTimer("Total Direct");
  traversal.direct(bodies, jbodies, cycle);
  logger::stopTimer("Total Direct");
  double potDif = verify.getDifScalar(bodies, bodies2);
  double potNrm = verify.getNrmScalar(bodies);
  double accDif = verify.getDifVector(bodies, bodies2);
  double accNrm = verify.getNrmVector(bodies);
  double potRel = std::sqrt(potDif/potNrm);
  double accRel = std::sqrt(accDif/accNrm);
  logger::printTitle("FMM vs. direct");
  verify.print("Rel. L2 Error (pot)",potRel);
  verify.print("Rel. L2 Error (acc)",accRel);
  buildTree.printTreeData(cells);
  traversal.printTraversalData();
  return 0;
}
