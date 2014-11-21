#include "args.h"
#include "dataset.h"
#include "logger.h"
#include "verify.h"

int main(int argc, char ** argv) {
  Args args(argc, argv);
  Bounds bounds;
  Dataset data;
  Verify verify;

  const real_t cycle = 2 * M_PI;
  logger::verbose = args.verbose;
  logger::printTitle("FMM Parameters");
  args.print(logger::stringLength, P);
  Bodies bodies = data.initBodies(args.numBodies, args.distribution, 0);
#if IneJ
  for (int b=0; b<int(bodies.size()); b++) {
    bodies[b].X[0] += M_PI;
    bodies[b].X[0] *= 0.5;
  }
  Bodies jbodies = data.initBodies(args.numBodies, args.distribution, 1);
  for (int b=0; b<int(jbodies.size()); b++) {
    bodies[b].X[0] -= M_PI;
    bodies[b].X[0] *= 0.5;
  }
#endif
  for (int it=0; it<args.repeat; it++) {
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();
    bounds = boundBox.getBounds(bodies);
#if IneJ
    bounds = boundBox.getBounds(jbodies,bounds);
#endif
    cells = buildTree.buildTree(bodies, bounds);
    upDownPass.upwardPass(cells);
#if IneJ
    jcells = buildTree.buildTree(jbodies, bounds);
    upDownPass.upwardPass(jcells);
    traversal.dualTreeTraversal(cells, jcells, cycle, false);
#else
    traversal.dualTreeTraversal(cells, cells, cycle, args.mutual);
    jbodies = bodies;
#endif
    upDownPass.downwardPass(cells);
    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");
#if WRITE_TIME
    logger::writeTime();
#endif
    const int numTargets = 100;
    bodies3 = bodies;
    data.sampleBodies(bodies, numTargets);
    bodies2 = bodies;
    data.initTarget(bodies);
    logger::startTimer("Total Direct");
    traversal.direct(bodies, jbodies, cycle);
    traversal.normalize(bodies);
    logger::stopTimer("Total Direct");
    double potDif = verify.getDifScalar(bodies, bodies2);
    double potNrm = verify.getNrmScalar(bodies);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);
    logger::printTitle("FMM vs. direct");
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
    buildTree.printTreeData(cells);
    traversal.printTraversalData();
    logger::printPAPI();
    logger::stopDAG();
    bodies = bodies3;
    data.initTarget(bodies);
  }
  logger::writeDAG();
#if VTK
  for (B_iter B=jbodies.begin(); B!=jbodies.end(); B++) B->IBODY = 0;
  for (C_iter C=jcells.begin(); C!=jcells.end(); C++) {
    Body body;
    body.IBODY = 1;
    body.X     = C->X;
    body.SRC   = 0;
    jbodies.push_back(body);
  }
  vtk3DPlot vtk;
  vtk.setBounds(M_PI,0);
  vtk.setGroupOfPoints(jbodies);
  vtk.plot();
#endif
  return 0;
}
