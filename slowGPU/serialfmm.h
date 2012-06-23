#ifndef serialfmm_h
#define serialfmm_h
#include "evaluator.h"

class SerialFMM : public Evaluator {
protected:
  int MAXLEVEL;

private:
  int getMaxLevel(Bodies &bodies) {
    const long N = bodies.size();
    int level;
    level = N >= NCRIT ? 1 + int(log(N / NCRIT)/M_LN2/3) : 0;
    return level;
  }

  inline void setMorton(Bodies &bodies) {
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
      int ix = int((B->X[0] + R0 - X0[0]) / d);
      int iy = int((B->X[1] + R0 - X0[1]) / d);
      int iz = int((B->X[2] + R0 - X0[2]) / d);
      int id = 0;
      for( int l=0; l!=MAXLEVEL; ++l ) {
        id += ix % 2 << (3 * l);
        id += iy % 2 << (3 * l + 1);
        id += iz % 2 << (3 * l + 2);
        ix >>= 1;
        iy >>= 1;
        iz >>= 1;
      }
      B->ICELL = id;
    }
  }

  inline void initCell(Cell &cell, int child, B_iter LEAF, real diameter) {
    cell.NCHILD = 0;
    cell.NCLEAF = 0;
    cell.NDLEAF = 0;
    cell.CHILD  = child;
    cell.LEAF   = LEAF;
    int ix = int((LEAF->X[0] + R0 - X0[0]) / diameter);
    int iy = int((LEAF->X[1] + R0 - X0[1]) / diameter);
    int iz = int((LEAF->X[2] + R0 - X0[2]) / diameter);
    cell.X[0]   = diameter * (ix + .5) + X0[0] - R0;
    cell.X[1]   = diameter * (iy + .5) + X0[1] - R0;
    cell.X[2]   = diameter * (iz + .5) + X0[2] - R0;
    cell.R      = diameter * .5;
  }

  void buildBottom(Bodies &bodies, Cells &cells) {
    int I = -1;
    Cell *C;
    cells.clear();
    cells.reserve(1 << (3 * MAXLEVEL));
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( B_iter B=bodies.begin(); B!=bodies.end(); ++B ) {
      int IC = B->ICELL;
      if( IC != I ) {
        Cell cell;
        initCell(cell,0,B,d);
        cell.ICELL = IC;
        cells.push_back(cell);
        C = &*cells.end()-1;
        I = IC;
      }
      C->NCLEAF++;
      C->NDLEAF++;
    }
  }

protected:
  void setDomain(Bodies &bodies) {
    MAXLEVEL = getMaxLevel(bodies);
    X0 = R0 = .5;
  }

  void buildTree(Bodies &bodies, Cells &cells) {
    setMorton(bodies);
    Bodies buffer = bodies;
    sort(bodies.begin(),bodies.end());
    buildBottom(bodies,cells);
  }

  void linkTree(Cells &cells) {
    int begin = 0, end = cells.size();
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int l=0; l!=MAXLEVEL; ++l ) {
      int div = (8 << (3 * l));
      int I = -1;
      int p = end - 1;
      d *= 2;
      for( int c=begin; c!=end; ++c ) {
        B_iter B = cells[c].LEAF;
        int IC = B->ICELL / div;
        if( IC != I ) {
          Cell cell;
          initCell(cell,c,cells[c].LEAF,d);
          cell.ICELL = IC;
          cells.push_back(cell);
          p++;
          I = IC;
        }
        cells[p].NCHILD++;
        cells[p].NDLEAF += cells[c].NDLEAF;
        cells[c].PARENT = p;
      }
      begin = end;
      end = cells.size();
    }
  }

public:
  void bottomup(Bodies &bodies, Cells &cells) {
    double tic, toc;
    tic = getTime();
    setDomain(bodies);
    buildTree(bodies,cells);
    linkTree(cells);
    toc = getTime();
    if( printNow ) printf("Tree                 : %lf\n",toc-tic);
    tic = getTime();
    Multipole = new real [cells.size()][MTERM]();
    Local = new real [cells.size()][LTERM]();
    upwardPass(cells);
    toc = getTime();
    if( printNow ) printf("Upward pass          : %lf\n",toc-tic);
  }

  void evaluate(Cells &icells, Cells &jcells) {
    double tic, toc;
    tic = getTime();
    setRootCell(icells,jcells);
    Pair pair(ROOT,ROOT2);
    PairQueue pairQueue;
    pairQueue.push_front(pair);
    traverse(pairQueue,false);
    toc = getTime();
    if( printNow ) printf("Traverse             : %lf\n",toc-tic);
    tic = getTime();
    downwardPass(icells);
    toc = getTime();
    if( printNow ) printf("Downward pass        : %lf\n",toc-tic);
    delete[] Multipole;
    delete[] Local;
  }

};

#endif
