#ifndef evaluator_h
#define evaluator_h
#include "cartesian.h"
#include "thread.h"
#define splitFirst(Ci,Cj) Cj->NCHILD == 0 || (Ci->NCHILD != 0 && Ci->RCRIT >= Cj->RCRIT)

class Evaluator : public Kernel {
private:
  int NM2L, NP2P;
  int NSPAWN;

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
  real_t getBmax(vec3 const&X, C_iter C) const {
    real_t rad = C->R;
    real_t dx = rad+std::abs(X[0]-C->X[0]);
    real_t dy = rad+std::abs(X[1]-C->X[1]);
    real_t dz = rad+std::abs(X[2]-C->X[2]);
    return std::sqrt( dx*dx + dy*dy + dz*dz );
  }

  void traverse(C_iter CiBegin, C_iter CiEnd, C_iter CjBegin, C_iter CjEnd, bool mutual) {
    if (CiEnd - CiBegin == 1 || CjEnd - CjBegin == 1) {
      if (CiBegin == CjBegin) {
        assert(CiEnd == CjEnd);
        traverse(CiBegin, CjBegin, mutual);
      } else {
        for (C_iter Ci=CiBegin; Ci!=CiEnd; Ci++) {
          for (C_iter Cj=CjBegin; Cj!=CjEnd; Cj++) {
            traverse(Ci, Cj, mutual);
          }
        }
      }
    } else {
      C_iter CiMid = CiBegin + (CiEnd - CiBegin) / 2;
      C_iter CjMid = CjBegin + (CjEnd - CjBegin) / 2;
      __init_tasks__;
      spawn_task0(traverse(CiBegin, CiMid, CjBegin, CjMid, mutual));
      traverse(CiMid, CiEnd, CjMid, CjEnd, mutual);
      __sync_tasks__;
      spawn_task0(traverse(CiBegin, CiMid, CjMid, CjEnd, mutual));
      if (!mutual || CiBegin != CjBegin) {
        traverse(CiMid, CiEnd, CjBegin, CjMid, mutual);
      } else {
        assert(CiEnd == CjEnd);
      }
      __sync_tasks__;
    }
  }

  void splitCell(C_iter Ci, C_iter Cj, bool mutual) {
    if (Cj->NCHILD == 0) {
      assert(Ci->NCHILD > 0);
      for (C_iter ci=Ci0+Ci->CHILD; ci!=Ci0+Ci->CHILD+Ci->NCHILD; ci++ ) {
        traverse(ci, Cj, mutual);
      }
    } else if (Ci->NCHILD == 0) {
      assert(Cj->NCHILD > 0);
      for (C_iter cj=Cj0+Cj->CHILD; cj!=Cj0+Cj->CHILD+Cj->NCHILD; cj++ ) {
        traverse(Ci, cj, mutual);
      }
    } else if (Ci->NDBODY + Cj->NDBODY >= NSPAWN || (mutual && Ci == Cj)) {
      traverse(Ci0+Ci->CHILD, Ci0+Ci->CHILD+Ci->NCHILD,
               Cj0+Cj->CHILD, Cj0+Cj->CHILD+Cj->NCHILD, mutual);
    } else if (Ci->RCRIT >= Cj->RCRIT) {
      for (C_iter ci=Ci0+Ci->CHILD; ci!=Ci0+Ci->CHILD+Ci->NCHILD; ci++ ) {
        traverse(ci, Cj, mutual);
      }
    } else {
      for (C_iter cj=Cj0+Cj->CHILD; cj!=Cj0+Cj->CHILD+Cj->NCHILD; cj++ ) {
        traverse(Ci, cj, mutual);
      }
    }
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
    real_t m = 0;
    vec3 X = 0;
    for( B_iter B=C->BODY; B!=C->BODY+C->NCBODY; ++B ) {
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

  void traverse(C_iter Ci, C_iter Cj, bool mutual) {
    vec3 dX = Ci->X - Cj->X - Xperiodic;                        // Distance vector from source to target
    real_t R2 = norm(dX);                                       // Scalar distance squared
#if DUAL
    {                                                           // Dummy bracket
#else
    if (Ci->RCRIT != Cj->RCRIT) {                               // If cell is not at the same level
      splitCell(Ci, Cj, mutual);                                //  Split cell and call function recursively for child
    } else {                                                    // If we don't care if cell is not at the same level
#endif
      if (R2 > (Ci->RCRIT+Cj->RCRIT)*(Ci->RCRIT+Cj->RCRIT)) {   //  If distance is far enough
        M2L(Ci, Cj, mutual);                                    //   Use approximate kernels
      } else if (Ci->NCHILD == 0 && Cj->NCHILD == 0) {          //  Else if both cells are bodies
        if (Cj->NCBODY == 0) {                                  //   If the bodies weren't sent from remote node
          M2L(Ci, Cj, mutual);                                  //    Use approximate kernels
        } else {                                                //   Else if the bodies were sent
          if (Ci == Cj) {                                       //    If source and target are same
            P2P(Ci);                                            //     P2P kernel for single cell
          } else {                                              //    Else if source and target are different
            P2P(Ci, Cj, mutual);                                //     P2P kernel for pair of cells
          }                                                     //    End if for same source and target
        }                                                       //   End if for bodies
      } else {                                                  //  Else if cells are close but not bodies
        splitCell(Ci, Cj, mutual);                              //   Split cell and call function recursively for child
      }                                                         //  End if for multipole acceptance
    }                                                           // End if for same level cells
  }

  void setRcrit(Cells &cells) {
#if ERROR_OPT
    real_t c = (1 - THETA) * (1 - THETA) / pow(THETA,P+2) / pow(std::abs(ROOT->M[0]),1.0/3);
#endif
    for( C_iter C=cells.begin(); C!=cells.end(); ++C ) {
      real_t x = 1.0 / THETA;
#if ERROR_OPT
      real_t a = c * pow(std::abs(C->M[0]),1.0/3);
      for( int i=0; i<5; ++i ) {
        real_t f = x * x - 2 * x + 1 - a * pow(x,-P);
        real_t df = (P + 2) * x - 2 * (P + 1) + P / x;
        x -= f / df;
      }
#endif
      C->RCRIT *= x;
    }
  }

public:
  Evaluator() : NM2L(0), NP2P(0), NSPAWN(1000), printNow(true) {}
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
      real_t Rmax = 0;
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
    Ci->BODY = ibodies.begin();
    Ci->NDBODY = ibodies.size();
    Cj->BODY = jbodies.begin();
    Cj->NDBODY = jbodies.size();
    for( B_iter B=ibodies.begin(); B!=ibodies.end(); ++B ) B->TRG = 0;
    P2P(Ci,Cj,false);
    real_t diff1 = 0, norm1 = 0, diff2 = 0, norm2 = 0;
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
