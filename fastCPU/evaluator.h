#ifndef evaluator_h
#define evaluator_h
#include "cartesian.h"
#define splitFirst(Ci,Cj) Cj->NCHILD == 0 || (Ci->NCHILD != 0 && Ci->RCRIT >= Cj->RCRIT)

class Evaluator : public Kernel {
private:
  int NM2L, NP2P;
protected:
  C_iter  ROOT, ROOT2;

public:
  bool printNow;

  double getTime() const {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return double(tv.tv_sec+tv.tv_usec*1e-6);
  }

private:
  real getBmax(vect const&X, C_iter C) const {
    real rad = C->R;
    real dx = rad+std::abs(X[0]-C->X[0]);
    real dy = rad+std::abs(X[1]-C->X[1]);
    real dz = rad+std::abs(X[2]-C->X[2]);
    return std::sqrt( dx*dx + dy*dy + dz*dz );
  }

  void interact(C_iter C, CellQueue &cellQueue) {
    if(C->NCHILD == 0 || C->NDLEAF < 64) {
      P2P(C);
      NP2P++;
    } else {
      cellQueue.push(C);
    }
  }

  void interact(C_iter Ci, C_iter Cj, PairQueue &pairQueue, bool mutual=true) {
    vect dX = Ci->X - Cj->X;
    real Rq = norm(dX);
#if DUAL
    if(Rq >= (Ci->RCRIT+Cj->RCRIT)*(Ci->RCRIT+Cj->RCRIT) && Rq != 0) {
      M2L(Ci,Cj,mutual);
      NM2L++;
    } else if(Ci->NCHILD == 0 && Cj->NCHILD == 0) {
      P2P(Ci,Cj,mutual);
      NP2P++;
    } else {
      Pair pair(Ci,Cj);
      pairQueue.push_back(pair);
    }
#else
    if(Ci->RCRIT != Cj->RCRIT) {
      Pair pair(Ci,Cj);
      pairQueue.push_back(pair);
    } else if(Rq >= (Ci->RCRIT+Cj->RCRIT)*(Ci->RCRIT+Cj->RCRIT) && Rq != 0) {
      M2L(Ci,Cj,mutual);
      NM2L++;
    } else if(Ci->NCHILD == 0 && Cj->NCHILD == 0) {
      P2P(Ci,Cj,mutual);
      NP2P++;
    } else {
      Pair pair(Ci,Cj);
      pairQueue.push_back(pair);
    }
#endif
  }

protected:
  void setRootCell(Cells &cells) {
    Ci0 = cells.begin();
    Cj0 = cells.begin();
    ROOT = cells.end() - 1;
  }

  void setRootCell(Cells &icells, Cells &jcells) {
    Ci0 = icells.begin();
    Cj0 = jcells.begin();
    ROOT  = icells.end() - 1;
    ROOT2 = jcells.end() - 1;
  }

  void setCenter(C_iter C) const {
    real m = 0;
    vect X = 0;
    for( B_iter B=C->LEAF; B!=C->LEAF+C->NCLEAF; ++B ) {
      m += B->SRC;
      X += B->X * B->SRC;
    }
    for( C_iter c=Cj0+C->CHILD; c!=Cj0+C->CHILD+C->NCHILD; ++c ) {
      m += std::abs(c->M[0]);
      X += c->X * std::abs(c->M[0]);
    }
    X /= m;
#if USE_BMAX
    C->R = getBmax(X,C);
#endif
#if COMcenter
    C->X = X;
#endif
  }

  void setRcrit(Cells &cells) {
#if ERROR_OPT
    real c = (1 - THETA) * (1 - THETA) / pow(THETA,P+2) / pow(std::abs(ROOT->M[0]),1.0/3);
#endif
    for( C_iter C=cells.begin(); C!=cells.end(); ++C ) {
      real x = 1.0 / THETA;
#if ERROR_OPT
      real a = c * pow(std::abs(C->M[0]),1.0/3);
      for( int i=0; i<5; ++i ) {
        real f = x * x - 2 * x + 1 - a * pow(x,-P);
        real df = (P + 2) * x - 2 * (P + 1) + P / x;
        x -= f / df;
      }
#endif
      C->RCRIT *= x;
    }
  }

  void traverse(PairQueue &pairQueue, bool mutual=true) {
    while( !pairQueue.empty() ) {
      Pair pair = pairQueue.front();
      pairQueue.pop_front();
      if(splitFirst(pair.first,pair.second)) {
        C_iter C = pair.first;
        for( C_iter Ci=Ci0+C->CHILD; Ci!=Ci0+C->CHILD+C->NCHILD; ++Ci ) {
          interact(Ci,pair.second,pairQueue,mutual);
        }
      } else {
        C_iter C = pair.second;
        for( C_iter Cj=Cj0+C->CHILD; Cj!=Cj0+C->CHILD+C->NCHILD; ++Cj ) {
          interact(pair.first,Cj,pairQueue,mutual);
        }
      }
    }
  }

  void traverse(CellQueue &cellQueue) {
    PairQueue pairQueue;
    while( !cellQueue.empty() ) {
      C_iter C = cellQueue.front();
      cellQueue.pop();
      for( C_iter Ci=Ci0+C->CHILD; Ci!=Ci0+C->CHILD+C->NCHILD; ++Ci ) {
        interact(Ci,cellQueue);
        for( C_iter Cj=Ci+1; Cj!=Cj0+C->CHILD+C->NCHILD; ++Cj ) {
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
    for( C_iter C=cells.begin(); C!=cells.end(); ++C ) {
      C->M = 0;
      C->L = 0;
    }
    for( C_iter C=cells.begin(); C!=cells.end(); ++C ) {
      real Rmax = 0;
      setCenter(C);
      P2M(C,Rmax);
      M2M(C,Rmax);
    }
    for( C_iter C=cells.begin(); C!=cells.end(); ++C ) {
      for( int i=1; i<MTERM; ++i ) C->M[i] /= C->M[0];
    }
    setRcrit(cells);
  }

  void downwardPass(Cells &cells) const {
    for( C_iter C=cells.end()-2; C!=cells.begin()-1; --C ) {
      L2L(C);
      L2P(C);
    }
  }

  void direct(Bodies &ibodies, Bodies &jbodies) {
    Cells cells;
    cells.resize(2);
    C_iter Ci = cells.begin(), Cj = cells.begin()+1;
    Ci->LEAF = ibodies.begin();
    Ci->NDLEAF = ibodies.size();
    Cj->LEAF = jbodies.begin();
    Cj->NDLEAF = jbodies.size();
    for( B_iter B=ibodies.begin(); B!=ibodies.end(); ++B ) B->TRG = 0;
    P2P(Ci,Cj,false);
    real diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
    B_iter B2=jbodies.begin();
    for( B_iter B=ibodies.begin(); B!=ibodies.end(); ++B, ++B2 ) {
      B->TRG /= B->SRC;
      diff1 += (B->TRG[0] - B2->TRG[0]) * (B->TRG[0] - B2->TRG[0]);// Difference of potential
      norm1 += B2->TRG[0] * B2->TRG[0];                         //  Value of potential
      diff2 += (B->TRG[1] - B2->TRG[1]) * (B->TRG[1] - B2->TRG[1]);// Difference of x acceleration
      diff2 += (B->TRG[2] - B2->TRG[2]) * (B->TRG[2] - B2->TRG[2]);// Difference of y acceleration
      diff2 += (B->TRG[3] - B2->TRG[3]) * (B->TRG[3] - B2->TRG[3]);// Difference of z acceleration
      norm2 += B2->TRG[1] * B2->TRG[1];                         //  Value of x acceleration
      norm2 += B2->TRG[2] * B2->TRG[2];                         //  Value of y acceleration
      norm2 += B2->TRG[3] * B2->TRG[3];
    }
    std::cout << std::setw(20) << std::left
              << "Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;
    std::cout << std::setw(20) << std::left
              << "Error (acc)" << " : " << std::sqrt(diff2/norm2) << std::endl;
  }
};

#undef splitFirst
#endif
