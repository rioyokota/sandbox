#include <fstream>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <sys/time.h>

const int N = 128*10000;
const int NCRIT = 10;
const int MAXLEVEL = N >= NCRIT ? 1 + int(log(N / NCRIT)/M_LN2/3) : 0;
const float THETA = 0.75;
const float EPS2 = 0.00001;
const float X0 = .5;
const float R0 = .5;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

struct Body {
  int ICELL;
  float X[3];
  float M;
  float P;
  float F[3];
};

struct Cell {
  int ICELL;
  int NCHILD;
  int NCLEAF;
  int NDLEAF;
  int CHILD;
  int LEAF;
  float X[3];
  float R;
  float M[7];
};

void getIndex(Body *bodies) {
  float diameter = 2 * R0 / (1 << MAXLEVEL);
  for( int i=0; i<N; i++ ) {
    int ix = (bodies[i].X[0] + R0 - X0) / diameter;
    int iy = (bodies[i].X[1] + R0 - X0) / diameter;
    int iz = (bodies[i].X[2] + R0 - X0) / diameter;
    int icell = 0;
    for( int l=0; l!=MAXLEVEL; ++l ) {
      icell += (ix & 1) << (3 * l);
      icell += (iy & 1) << (3 * l + 1);
      icell += (iz & 1) << (3 * l + 2);
      ix >>= 1;
      iy >>= 1;
      iz >>= 1;
    }
    bodies[i].ICELL = icell;
  }
}

void sortBody(Body *bodies, Body *buffer) {
  int Imin = bodies[0].ICELL;
  int Imax = bodies[0].ICELL;
  for( int i=0; i<N; ++i ) {
    if     ( bodies[i].ICELL < Imin ) Imin = bodies[i].ICELL;
    else if( bodies[i].ICELL > Imax ) Imax = bodies[i].ICELL;
  }
  int numBucket = Imax - Imin + 1;
  int *bucket = new int [numBucket];
  for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
  for( int i=0; i!=N; i++ ) bucket[bodies[i].ICELL-Imin]++;
  for( int i=1; i!=numBucket; i++ ) bucket[i] += bucket[i-1];
  for( int i=N-1; i>=0; i-- ) {
    bucket[bodies[i].ICELL-Imin]--;
    int inew = bucket[bodies[i].ICELL-Imin];
    buffer[inew] = bodies[i];
  }
  for( int i=0; i<N; i++ ) bodies[i] = buffer[i];
  delete[] bucket;
}

void initCell(const Body *bodies, Cell &cell, const int &icell, const int &leaf, const float &diameter) {
  cell.ICELL = icell;
  cell.NCHILD = 0;
  cell.NCLEAF = 0;
  cell.NDLEAF = 0;
  cell.LEAF   = leaf;
  int ix = (bodies[leaf].X[0] + R0 - X0) / diameter;
  int iy = (bodies[leaf].X[1] + R0 - X0) / diameter;
  int iz = (bodies[leaf].X[2] + R0 - X0) / diameter;
  cell.X[0]   = diameter * (ix + .5) + X0 - R0;
  cell.X[1]   = diameter * (iy + .5) + X0 - R0;
  cell.X[2]   = diameter * (iz + .5) + X0 - R0;
  cell.R      = diameter * .5;
  for( int i=0; i<7; i++ ) cell.M[i] = 0;
}

void buildCell(const Body *bodies, Cell *cells, int &ncell) {
  int oldcell = -1;
  ncell = 0;
  float diameter = 2 * R0 / (1 << MAXLEVEL);
  for( int i=0; i<N; i++ ) {
    int icell = bodies[i].ICELL;
    if( icell != oldcell ) {
      initCell(bodies,cells[ncell],icell,i,diameter);
      oldcell = icell;
      ncell++;
    }
    cells[ncell-1].NCLEAF++;
    cells[ncell-1].NDLEAF++;
  }
}

void buildTree(const Body *bodies, Cell *cells, int &ncell) {
  int begin = 0, end = ncell;
  float diameter = 2 * R0 / (1 << MAXLEVEL);
  for( int level=MAXLEVEL-1; level>=0; level-- ) {
    int oldcell = -1;
    diameter *= 2;
    for( int i=begin; i!=end; ++i ) {
      int icell = cells[i].ICELL / 8;
      if( icell != oldcell ) {
        initCell(bodies,cells[ncell],icell,cells[i].LEAF,diameter);
        cells[ncell].CHILD = i;
        oldcell = icell;
        ncell++;
      }
      cells[ncell-1].NCHILD++;
      cells[ncell-1].NDLEAF += cells[i].NDLEAF;
    }
    begin = end;
    end = ncell;
  }
}

float getBmax(const float *X, const Cell &cell) {
  float dx = cell.R + fabs(X[0] - cell.X[0]);
  float dy = cell.R + fabs(X[1] - cell.X[1]);
  float dz = cell.R + fabs(X[2] - cell.X[2]);
  return sqrtf( dx*dx + dy*dy + dz*dz );
}

void setCenter(Body *bodies, Cell *cells, Cell &cell) {
  float M = 0;
  float X[3] = {0,0,0};
  for( int i=0; i<cell.NCLEAF; i++ ) {
    Body body = bodies[cell.LEAF+i];
    M += body.M;
    X[0] += body.X[0] * body.M;
    X[1] += body.X[1] * body.M;
    X[2] += body.X[2] * body.M;
  }
  for( int c=0; c<cell.NCHILD; c++ ) {
    Cell child = cells[cell.CHILD+c];
    M += fabs(child.M[0]);
    X[0] += child.X[0] * fabs(child.M[0]);
    X[1] += child.X[1] * fabs(child.M[0]);
    X[2] += child.X[2] * fabs(child.M[0]);
  }
  X[0] /= M;
  X[1] /= M;
  X[2] /= M;
  cell.R = getBmax(X,cell);
  cell.X[0] = X[0];
  cell.X[1] = X[1];
  cell.X[2] = X[2];
}

inline void P2M(Cell &cell, const Body *bodies) {
  for( int i=0; i<cell.NCLEAF; i++ ) {
    Body leaf = bodies[cell.LEAF+i];
    float dx = cell.X[0]-leaf.X[0];
    float dy = cell.X[1]-leaf.X[1];
    float dz = cell.X[2]-leaf.X[2];
    cell.M[0] += leaf.M;
    cell.M[1] += leaf.M * dx * dx / 2;
    cell.M[2] += leaf.M * dx * dy / 2;
    cell.M[3] += leaf.M * dx * dz / 2;
    cell.M[4] += leaf.M * dy * dy / 2;
    cell.M[5] += leaf.M * dy * dz / 2;
    cell.M[6] += leaf.M * dz * dz / 2;
  }
}

inline void M2M(Cell &parent, const Cell *cells) {
  for( int c=0; c<parent.NCHILD; c++ ) {
    Cell child = cells[parent.CHILD+c];
    float dx = parent.X[0] - child.X[0];
    float dy = parent.X[1] - child.X[1];
    float dz = parent.X[2] - child.X[2];
    parent.M[0] += child.M[0];
    parent.M[1] += child.M[1] + dx * dx * child.M[0] / 2;
    parent.M[2] += child.M[2];
    parent.M[3] += child.M[3];
    parent.M[4] += child.M[4] + dy * dy * child.M[0] / 2;
    parent.M[5] += child.M[5];
    parent.M[6] += child.M[6] + dz * dz * child.M[0] / 2;
  }
}

inline void P2P(Body *bodies, const Cell &icell, const Cell &jcell) {
  for( int i=0; i<icell.NDLEAF; i++ ) {
    Body *ileaf = &bodies[icell.LEAF+i];
    for( int j=0; j<jcell.NDLEAF; j++ ) {
      Body jleaf = bodies[jcell.LEAF+j];
      float dx = ileaf->X[0] - jleaf.X[0];
      float dy = ileaf->X[1] - jleaf.X[1];
      float dz = ileaf->X[2] - jleaf.X[2];
      float invR = 1 / sqrtf(dx * dx + dy * dy + dz * dz + EPS2);
      float invR3 = -jleaf.M * invR * invR * invR;
      ileaf->P += jleaf.M * invR;
      ileaf->F[0] += dx * invR3;
      ileaf->F[1] += dy * invR3;
      ileaf->F[2] += dz * invR3;
    }
  }
}

inline void M2P(Body *bodies, const Cell &icell, const Cell &jcell) {
  for( int i=0; i<icell.NDLEAF; i++ ) {
    Body *ileaf = &bodies[icell.LEAF+i];
    float dx = ileaf->X[0] - jcell.X[0];
    float dy = ileaf->X[1] - jcell.X[1];
    float dz = ileaf->X[2] - jcell.X[2];
    float invR = 1 / sqrtf(dx * dx + dy * dy + dz * dz);
    float invR3 = -invR * invR * invR;
    float invR5 = -3 * invR3 * invR * invR;
    ileaf->P += jcell.M[0] * invR;
    ileaf->P += jcell.M[1] * (dx * dx * invR5 + invR3);
    ileaf->P += jcell.M[2] * dx * dy * invR5;
    ileaf->P += jcell.M[3] * dx * dz * invR5;
    ileaf->P += jcell.M[4] * (dy * dy * invR5 + invR3);
    ileaf->P += jcell.M[5] * dy * dz * invR5;
    ileaf->P += jcell.M[6] * (dz * dz * invR5 + invR3);
    ileaf->F[0] += jcell.M[0] * dx * invR3;
    ileaf->F[1] += jcell.M[0] * dy * invR3;
    ileaf->F[2] += jcell.M[0] * dz * invR3;
  }
}

int main() {
  Body *bodies = new Body [N];
  Body *bodies2 = new Body [N];
// Initialize
  std::cout << "N     : " << N << std::endl << std::endl;
  for( int i=0; i!=N; ++i ) {
    bodies[i].X[0] = drand48();
    bodies[i].X[1] = drand48();
    bodies[i].X[2] = drand48();
    bodies[i].M = 1. / N;
    bodies[i].P = -bodies[i].M / sqrtf(EPS2);
    bodies[i].F[0] = bodies[i].F[1] = bodies[i].F[2] = 0;
  }
// Set root cell
  double tic = get_time();
// Build tree
  getIndex(bodies);
  Body *buffer = new Body [N];
  sortBody(bodies,buffer);
  delete[] buffer; 
  Cell *cells = new Cell [N];
  int ncell = 0;
  buildCell(bodies,cells,ncell);
  double toc = get_time();
  std::cout << "Index : " << toc-tic << std::endl;
  tic = get_time();
  int ntwig = ncell;
  buildTree(bodies,cells,ncell);
// Upward sweep
  for( int i=0; i<ncell-1; i++ ) {
    setCenter(bodies,cells,cells[i]);
    P2M(cells[i],bodies);
    M2M(cells[i],cells);
  }
  toc = get_time();
  std::cout << "Build : " << toc-tic << std::endl;
// Direct summation
  tic = get_time();
#if 0
  Cell root = cells[ncell-1];
  P2P(bodies,root,root);
  for( int i=0; i<N; i++ ) {
    bodies2[i].P = bodies[i].P;
    bodies2[i].F[0] = bodies[i].F[0];
    bodies2[i].F[1] = bodies[i].F[1];
    bodies2[i].F[2] = bodies[i].F[2];
    bodies[i].P = -bodies[i].M / sqrtf(EPS2);
    bodies[i].F[0] = bodies[i].F[1] = bodies[i].F[2] = 0;
  }
#endif
  toc = get_time();
  std::cout << "Direct: " << toc-tic << std::endl;
// Evaluate
  tic = get_time();
  int NP2P = 0, NM2P = 0;
  int stack[20];
  int nstack = 0;
  for( int i=0; i<ntwig; i++ ) {
    Cell icell = cells[i];
    stack[nstack++] = ncell-1;
    int nP2P = 0;
    while( nstack ) {
      Cell jparent = cells[stack[--nstack]];
      for( int j=0; j<jparent.NCHILD; j++ ) {
        Cell jcell = cells[jparent.CHILD+j];
        float dx = icell.X[0] - jcell.X[0];
        float dy = icell.X[1] - jcell.X[1];
        float dz = icell.X[2] - jcell.X[2];
        float R = sqrtf(dx * dx + dy * dy + dz * dz);
        if( jcell.R < THETA * R ) {
          M2P(bodies,icell,jcell);
          NM2P++;
        } else if( jcell.NCHILD == 0 ) {
          P2P(bodies,icell,jcell);
          NP2P++;
          nP2P++;
        } else {
          stack[nstack++] = jparent.CHILD+j;
        }
      }
    }
  }
  toc = get_time();
  std::cout << "FMM   : " << toc-tic << std::endl << std::endl;
  std::cout << "NP2P  : " << NP2P << std::endl;
  std::cout << "NM2P  : " << NM2P << std::endl << std::endl;
// Check accuracy
  float errp = 0, relp = 0, errf = 0, relf = 0;
  std::ifstream fid("direct");
  for( int i=0; i<N; i++ ) {
    Body body = bodies[i];
    Body body2 = bodies2[i];
    fid >> body2.P >> body2.F[0] >> body2.F[1] >> body2.F[2];
    errp += (body2.P - body.P) * (body2.P - body.P);
    relp += body2.P * body2.P;
    errf += (body2.F[0] - body.F[0]) * (body2.F[0] - body.F[0])
          + (body2.F[1] - body.F[1]) * (body2.F[1] - body.F[1])
          + (body2.F[2] - body.F[2]) * (body2.F[2] - body.F[2]);
    relf += body2.F[0] * body2.F[0] + body2.F[1] * body2.F[1] + body2.F[2] * body2.F[2];
  }
  fid.close();
  std::cout << "P err : " << sqrtf(errp/relp) << std::endl;
  std::cout << "F err : " << sqrtf(errf/relf) << std::endl;
  delete[] bodies;
  delete[] bodies2;
  delete[] cells;
}
