#ifndef evaluator_h
#define evaluator_h
#include "kernel.h"
#define splitFirst(Ci,Cj) Cj->NCHILD == 0 || (Ci->NCHILD != 0 && Ci->RCRIT >= Cj->RCRIT)

class Evaluator : public Kernel {
private:
  int NM2L, NP2P;
protected:
  int numBodies;
  int numCells;

public:
  bool printNow;

  double getTime() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return double(tv.tv_sec+tv.tv_usec*1e-6);
  }

private:
  real getBmax(vec3 const&X, Cell *C) const {
    real rad = C->R;
    real dx = rad+std::abs(X[0]-C->X[0]);
    real dy = rad+std::abs(X[1]-C->X[1]);
    real dz = rad+std::abs(X[2]-C->X[2]);
    return std::sqrt( dx*dx + dy*dy + dz*dz );
  }

  void interact(Cell *Ci, Cell *Cj, PairStack &pairStack) {
    vec3 dX = Ci->X - Cj->X;
    real Rq = norm(dX);
    if(Rq >= (Ci->RCRIT+Cj->RCRIT)*(Ci->RCRIT+Cj->RCRIT) && Rq != 0) {
      M2L(Ci,Cj);
      NM2L++;
    } else if(Ci->NCHILD == 0 && Cj->NCHILD == 0) {
      P2P(Ci,Cj);
      NP2P++;
    } else {
      CellPair pair(Ci,Cj);
      pairStack.push(pair);
    }
  }

protected:
  void setCenter(Cell *C) const {
    real m = 0;
    vec3 X = 0;
    for( int b=C->LEAF; b<C->LEAF+C->NCLEAF; ++b ) {
      m += Jbodies[b][3];
      for( int d=0; d<3; d++ ) X[d] += Jbodies[b][d] * Jbodies[b][3];
    }
    for( int c=C->CHILD; c<C->CHILD+C->NCHILD; c++ ) {
      Cell *CC = Cells.host() + c;
      m += std::abs(Multipole[c][0]);
      X += CC->X * std::abs(Multipole[c][0]);
    }
    X /= m;
    C->R = getBmax(X,C);
    C->X = X;
  }

  void setRcrit() {
    real coef = (1 - THETA) * (1 - THETA) / pow(THETA,P+2) / pow(std::abs(Multipole[numCells-1][0]),1.0/3);
    for( int c=0; c<numCells; ++c ) {
      Cell *C = Cells.host() + c;
      real x = 1.0 / THETA;
      real a = coef * pow(std::abs(Multipole[c][0]),1.0/3);
      for( int i=0; i<5; ++i ) {
        real f = x * x - 2 * x + 1 - a * pow(x,-P);
        real df = (P + 2) * x - 2 * (P + 1) + P / x;
        x -= f / df;
      }
      C->RCRIT *= x;
    }
  }

  void traverse() {
    Cell *root = Cells.host() + numCells - 1;
    CellPair pair(root,root);
    PairStack pairStack;
    pairStack.push(pair);
    while( !pairStack.empty() ) {
      CellPair pair = pairStack.pop();
      if(splitFirst(pair.first,pair.second)) {
        Cell *C = pair.first;
        for( Cell *Ci=Cells.host()+C->CHILD; Ci<Cells.host()+C->CHILD+C->NCHILD; ++Ci ) {
          interact(Ci,pair.second,pairStack);
        }
      } else {
        Cell *C = pair.second;
        for( Cell *Cj=Cells.host()+C->CHILD; Cj<Cells.host()+C->CHILD+C->NCHILD; ++Cj ) {
          interact(pair.first,Cj,pairStack);
        }
      }
    }
  }

public:
  Evaluator() : NM2L(0), NP2P(0), printNow(true) {}
  ~Evaluator() {
    std::cout << "NM2L : " << NM2L << " NP2P : " << NP2P << std::endl;
  }

  void upwardPass() {
    for( int c=0; c<numCells; ++c ) {
      Multipole[c] = 0;
      Local[c] = 0;
    }
    for( int c=0; c<numCells; ++c ) {
      Cell *C = Cells.host() + c;
      setCenter(C);
      P2M(C);
      M2M(C);
    }
    for( int c=0; c<numCells; ++c ) {
      for( int i=1; i<MTERM; ++i ) Multipole[c][i] /= Multipole[c][0];
    }
    setRcrit();
  }

  void downwardPass() const {
    for( int c=numCells-2; c>=0; --c ) {
      Cell *C = Cells.host() + c;
      L2L(C);
      L2P(C);
    }
  }

  void direct() {
    Cell Ci[2];
    Cell *Cj = Ci + 1;
    Ci->LEAF = 0;
    Ci->NDLEAF = 100;
    Cj->LEAF = 0;
    Cj->NDLEAF = numBodies;
    vec4 *Ibodies2 = new vec4 [100];
    for( int b=0; b<100; b++ ) {
      for( int d=0; d<4; d++ ) {
        Ibodies2[b][d] = Ibodies[b][d];
        Ibodies[b][d] = 0;
      }
    }
    P2P(Ci,Cj);
    real diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
    for( int b=0; b<100; ++b ) {
      Ibodies[b] /= Jbodies[b][3];
      diff1 += (Ibodies[b][0] - Ibodies2[b][0]) * (Ibodies[b][0] - Ibodies2[b][0]);
      norm1 += Ibodies[b][0] * Ibodies[b][0];
      diff2 += (Ibodies[b][1] - Ibodies2[b][1]) * (Ibodies[b][1] - Ibodies2[b][1]);
      diff2 += (Ibodies[b][2] - Ibodies2[b][2]) * (Ibodies[b][2] - Ibodies2[b][2]);
      diff2 += (Ibodies[b][3] - Ibodies2[b][3]) * (Ibodies[b][3] - Ibodies2[b][3]);
      norm2 += Ibodies[b][1] * Ibodies[b][1];
      norm2 += Ibodies[b][2] * Ibodies[b][2];
      norm2 += Ibodies[b][3] * Ibodies[b][3];
    }
    std::cout << std::setw(20) << std::left
              << "Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;
    std::cout << std::setw(20) << std::left
              << "Error (acc)" << " : " << std::sqrt(diff2/norm2) << std::endl;
    delete[] Ibodies2;
  }
};

#undef splitFirst
#endif
