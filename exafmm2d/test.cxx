#include <cassert>

#include "boundbox.h"
#include "buildtree.h"
#include "dataset.h"
#include "logger.h"
#include "traversal.h"
#include "updownpass.h"

int main(int argc, char ** argv) {
  Dataset data;
  Logger logger;

  const int numBodies = 1000000;
  const int images = 0;
  const int ncrit = 8;
  const int nspawn = 1000;
  const real_t theta = 0.4;
  const real_t eps2 = 0.0;
  const real_t cycle = 2 * M_PI;
  BoundBox boundbox(nspawn);
  BuildTree tree(ncrit,nspawn);
  UpDownPass pass(theta,eps2);
  Traversal traversal(nspawn,images,eps2);
  logger.printTitle("FMM Profiling");
  logger.startTimer("Total FMM");
  Bodies bodies = data.initBodies(numBodies, 0);
  Bounds bounds = boundbox.getBounds(bodies);
  Cells cells = tree.buildTree(bodies, bounds);
  pass.upwardPass(cells);
  traversal.dualTreeTraversal(cells, cells, cycle);
  Bodies jbodies = bodies;
  pass.downwardPass(cells);
  logger.printTitle("Total runtime");
  logger.stopTimer("Total FMM");
  data.sampleBodies(bodies, 10);
  Bodies bodies2 = bodies;
  data.initTarget(bodies);
  logger.startTimer("Total Direct");
  traversal.direct(bodies, jbodies, cycle);
  traversal.normalize(bodies);
  logger.stopTimer("Total Direct");
  double diff1 = 0, norm1 = 0;
  data.evalError(bodies2, bodies, diff1, norm1);
  logger.printTitle("FMM vs. direct");
  logger.printError(diff1, norm1);
  return 0;
}
