#ifndef evaluator_h
#define evaluator_h
#include "kernel.h"
#define splitFirst(Ci,Cj) Cj->NCHILD == 0 || (Ci->NCHILD != 0 && Ci->RCRIT >= Cj->RCRIT)

class Evaluator : public Kernel {
private:
  int NM2L, NP2P;
protected:
  Cell *ROOT;

public:
  bool printNow;

  double getTime() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return double(tv.tv_sec+tv.tv_usec*1e-6);
  }

private:
  real getBmax(vect const&X, Cell *C) const {
    real rad = C->R;
    real dx = rad+std::abs(X[0]-C->X[0]);
    real dy = rad+std::abs(X[1]-C->X[1]);
    real dz = rad+std::abs(X[2]-C->X[2]);
    return std::sqrt( dx*dx + dy*dy + dz*dz );
  }

  void interact(Cell *C, CellQueue &cellQueue) {
    if(C->NCHILD == 0 || C->NDLEAF < 64) {
      P2P(C);
      NP2P++;
    } else {
      cellQueue.push(C);
    }
  }

  void interact(Cell *Ci, Cell *Cj, PairQueue &pairQueue) {
    vect dX = Ci->X - Cj->X;
    real Rq = norm(dX);
    if(Rq >= (Ci->RCRIT+Cj->RCRIT)*(Ci->RCRIT+Cj->RCRIT) && Rq != 0) {
      M2L(Ci,Cj);
      NM2L++;
    } else if(Ci->NCHILD == 0 && Cj->NCHILD == 0) {
      P2P(Ci,Cj);
      NP2P++;
    } else {
      Pair pair(Ci,Cj);
      pairQueue.push_back(pair);
    }
  }

protected:
  void setRootCell(Cells &cells) {
    C0 = &cells.front();
    ROOT = &cells.back();
  }

  void setCenter(Cell *C) const {
    real m = 0;
    vect X = 0;
    for( Body *B=B0+C->LEAF; B<B0+C->LEAF+C->NCLEAF; ++B ) {
      m += Jbodies[B-B0][3];
      X += B->X * B->SRC;
    }
    for( Cell *c=C0+C->CHILD; c<C0+C->CHILD+C->NCHILD; ++c ) {
      m += std::abs(Multipole[c-C0][0]);
      X += c->X * std::abs(Multipole[c-C0][0]);
    }
    X /= m;
    C->R = getBmax(X,C);
    C->X = X;
  }

  void setRcrit(Cells &cells) {
    real c = (1 - THETA) * (1 - THETA) / pow(THETA,P+2) / pow(std::abs(Multipole[ROOT-C0][0]),1.0/3);
    for( Cell *C=&*cells.begin(); C<&*cells.end(); ++C ) {
      real x = 1.0 / THETA;
      real a = c * pow(std::abs(Multipole[C-C0][0]),1.0/3);
      for( int i=0; i<5; ++i ) {
        real f = x * x - 2 * x + 1 - a * pow(x,-P);
        real df = (P + 2) * x - 2 * (P + 1) + P / x;
        x -= f / df;
      }
      C->RCRIT *= x;
    }
  }

  void traverse(PairQueue &pairQueue) {
    while( !pairQueue.empty() ) {
      Pair pair = pairQueue.front();
      pairQueue.pop_front();
      if(splitFirst(pair.first,pair.second)) {
        Cell *C = pair.first;
        for( Cell *Ci=C0+C->CHILD; Ci<C0+C->CHILD+C->NCHILD; ++Ci ) {
          interact(Ci,pair.second,pairQueue);
        }
      } else {
        Cell *C = pair.second;
        for( Cell *Cj=C0+C->CHILD; Cj<C0+C->CHILD+C->NCHILD; ++Cj ) {
          interact(pair.first,Cj,pairQueue);
        }
      }
    }
  }

  void traverse(CellQueue &cellQueue) {
    PairQueue pairQueue;
    while( !cellQueue.empty() ) {
      Cell *C = cellQueue.front();
      cellQueue.pop();
      for( Cell *Ci=C0+C->CHILD; Ci<C0+C->CHILD+C->NCHILD; ++Ci ) {
        interact(Ci,cellQueue);
        for( Cell *Cj=Ci+1; Cj<C0+C->CHILD+C->NCHILD; ++Cj ) {
          interact(Ci,Cj,pairQueue);
        }
      }
      traverse(pairQueue);
    }
  }

public:
  Evaluator() : NM2L(0), NP2P(0), printNow(true) {}
  ~Evaluator() {
    std::cout << "NM2L : " << NM2L << " NP2P : " << NP2P << std::endl;
  }

  void upwardPass(Cells &cells) {
    setRootCell(cells);
    int c = 0;
    for( Cell *C=&*cells.begin(); C<&*cells.end(); ++C,++c ) {
      for( int i=0; i<MTERM; ++i ) Multipole[c][i] = 0;
      for( int i=0; i<LTERM; ++i ) Local[c][i] = 0;
    }
    for( Cell *C=&*cells.begin(); C<&*cells.end(); ++C ) {
      real Rmax = 0;
      setCenter(C);
      P2M(C,Rmax);
      M2M(C,Rmax);
    }
    c = 0;
    for( Cell *C=&*cells.begin(); C<&*cells.end(); ++C,++c ) {
      for( int i=1; i<MTERM; ++i ) Multipole[c][i] /= Multipole[c][0];
    }
    setRcrit(cells);
  }

  void downwardPass(Cells &cells) const {
    for( Cell *C=&*cells.end()-2; C>=&*cells.begin(); --C ) {
      L2L(C);
      L2P(C);
    }
  }

  void direct(Bodies &jbodies) {
    Cell Ci[2];
    Cell *Cj = Ci + 1;
    Bodies ibodies = jbodies;
    B0 = &ibodies.front();
    Ci->LEAF = 0;
    Ci->NDLEAF = 100;
    Cj->LEAF = 0;
    Cj->NDLEAF = ibodies.size();
    for( int b=0; b<100; ++b ) ibodies[b].TRG = 0;
    P2P(Ci,Cj);
    real diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
    for( int b=0; b<100; ++b ) {
      Body *B = &ibodies[b];
      Body *B2 = &jbodies[b];
      B->TRG /= B->SRC;
      diff1 += (B->TRG[0] - B2->TRG[0]) * (B->TRG[0] - B2->TRG[0]);
      norm1 += B->TRG[0] * B->TRG[0];
      diff2 += (B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]);
      diff2 += (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]);
      diff2 += (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]);
      norm2 += B->TRG[1] * B->TRG[1];
      norm2 += B->TRG[2] * B->TRG[2];
      norm2 += B->TRG[3] * B->TRG[3];
    }
    std::cout << std::setw(20) << std::left
              << "Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;
    std::cout << std::setw(20) << std::left
              << "Error (acc)" << " : " << std::sqrt(diff2/norm2) << std::endl;
  }
};

#undef splitFirst
#endif
