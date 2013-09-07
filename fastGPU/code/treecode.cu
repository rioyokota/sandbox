#include "Treecode.h"

int main(int argc, char * argv[])
{
  std::string fileName = "";
  int seed = 19810614;
  int nPtcl = -1;
  {
    AnyOption opt;
#define ADDUSAGE(line) {{std::stringstream oss; oss << line; opt.addUsage(oss.str());}}
		ADDUSAGE(" ");
		ADDUSAGE("Usage");
		ADDUSAGE(" ");
		ADDUSAGE(" -h  --help             Prints this help ");
		ADDUSAGE(" -i  --infile #         Input snapshot filename [tipsy format]");
    ADDUSAGE(" -n  --plummer #        Generate plummer model with a given number of particles");
    ADDUSAGE(" -s  --seed    #        Random seed [" << seed << "]"); 
		ADDUSAGE(" ");
#undef  ADDUSAGE

    opt.setFlag( "help" ,   'h');
    opt.setOption( "infile",  'i');
		opt.setOption( "plummer", 'n' );
		opt.setOption( "seed", 's' );
		
    opt.processCommandArgs( argc, argv );

		if( ! opt.hasOptions() || opt.getFlag("help") || opt.getFlag('h')) 
    {
			opt.printUsage();
			exit(0);
		}
		
		char *optarg = NULL;
    if ((optarg = opt.getValue("plummer"))) nPtcl = atoi(optarg);
    if ((optarg = opt.getValue("seed")))    seed = atoi(optarg);
    if ((optarg = opt.getValue("infile")))  fileName = std::string(optarg);
  }

  typedef float real_t;
  typedef Treecode<real_t> Tree;

  const real_t eps   = 0.05;
  const real_t theta = 0.75;
  Tree tree(eps, theta);

  fprintf(stdout, "Using Plummer model with nPtcl= %d\n", nPtcl);
  const Plummer data(nPtcl, seed);
  tree.alloc(nPtcl);
  for (int i = 0; i < nPtcl; i++) {
    typename Tree::Particle ptclPos, ptclVel, ptclAcc;
    ptclPos.x()    = data.pos[i].x;
    ptclPos.y()    = data.pos[i].y;
    ptclPos.z()    = data.pos[i].z;
    ptclPos.mass() = data.mass[i];

    ptclVel.x()    = data.vel[i].x;
    ptclVel.y()    = data.vel[i].y;
    ptclVel.z()    = data.vel[i].z;
    ptclVel.mass() = data.mass[i];

    ptclAcc.x()    = 0;
    ptclAcc.y()    = 0;
    ptclAcc.z()    = 0;
    ptclAcc.mass() = 0;

    tree.h_ptclPos[i] = ptclPos;
    tree.h_ptclVel[i] = ptclVel;
    tree.h_ptclAcc[i] = ptclAcc;
    tree.h_ptclAcc2[i] = make_float4(0,0,0,0);
  }

#if 1
  {
    double mtot = 0.0;
    typename vec<3,real_t>::type bmin = {+1e10};
    typename vec<3,real_t>::type bmax = {-1e10};
    for (int i = 0; i < nPtcl; i++)
    {
      const Tree::Particle pos = tree.h_ptclPos[i];
      mtot += pos.mass();
      bmin.x = std::min(bmin.x, pos.x());
      bmin.y = std::min(bmin.y, pos.y());
      bmin.z = std::min(bmin.z, pos.z());
      bmax.x = std::max(bmax.x, pos.x());
      bmax.y = std::max(bmax.y, pos.y());
      bmax.z = std::max(bmax.z, pos.z());
    }
    fprintf(stderr, " Total mass = %g \n", mtot);
    fprintf(stderr, "  bmin= %g %g %g \n", bmin.x, bmin.y, bmin.z);
    fprintf(stderr, "  bmax= %g %g %g \n", bmax.x, bmax.y, bmax.z);
  }
#endif

  tree.ptcl_h2d();

  double t0 = rtc();
  tree.buildTree(64);          /* pass nLeaf, accepted 16, 24, 32, 48, 64 */
  tree.computeMultipoles();
  tree.makeGroups(5, 64);     /* pass nCrit */
#if 1
  for (int k = 0; k < 1; k++)
  {
    t0 = rtc();
    const double4 interactions = tree.computeForces();
    double dt = rtc() - t0;
#ifdef QUADRUPOLE
    const int FLOPS_QUAD = 64;
#else
    const int FLOPS_QUAD = 20;
#endif

    fprintf(stderr, " direct: <n>= %g  max= %g  approx: <n>=%g max= %g :: perf= %g TFLOP/s \n",
        interactions.x, interactions.y, 
        interactions.z, interactions.w,
        (interactions.x*20 + interactions.z*FLOPS_QUAD)*tree.get_nPtcl()/dt/1e12);

  }
#else
  tree.computeForces();
#endif
  fprintf(stderr,"direct summation ...\n");
  const int numTarget = 512; // Number of threads per block will be set to this value
  const int numBlock = 64;
  t0 = rtc();
  tree.computeDirect(numTarget,numBlock);
  double dt = rtc() - t0;
  fprintf(stderr,"time= %g  perf= %g TFLOP/S\n",dt,20.*numTarget*nPtcl/dt/1e12);
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
    diffp += (tree.h_ptclAcc[i].mass() - tree.h_ptclAcc2[i].w) * (tree.h_ptclAcc[i].mass() - tree.h_ptclAcc2[i].w);
    diffa += (tree.h_ptclAcc[i].x() - tree.h_ptclAcc2[i].x) * (tree.h_ptclAcc[i].x() - tree.h_ptclAcc2[i].x)
      + (tree.h_ptclAcc[i].y() - tree.h_ptclAcc2[i].y) * (tree.h_ptclAcc[i].y() - tree.h_ptclAcc2[i].y)
      + (tree.h_ptclAcc[i].z() - tree.h_ptclAcc2[i].z) * (tree.h_ptclAcc[i].z() - tree.h_ptclAcc2[i].z);
    normp += tree.h_ptclAcc2[i].w * tree.h_ptclAcc2[i].w;
    norma += tree.h_ptclAcc2[i].x * tree.h_ptclAcc2[i].x
      + tree.h_ptclAcc2[i].y * tree.h_ptclAcc2[i].y
      + tree.h_ptclAcc2[i].z * tree.h_ptclAcc2[i].z;
  }
  printf("pot : %g\n",sqrt(diffp/normp));
  printf("acc : %g\n",sqrt(diffa/norma));
  fprintf(stderr, " nLeaf= %d  nCrit= %d\n", tree.get_nLeaf(), tree.get_nCrit());


  return 0;

}
