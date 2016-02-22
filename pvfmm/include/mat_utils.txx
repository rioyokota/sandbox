namespace pvfmm{
namespace mat{

  template <class T>
  void tpinv(T* M, int n1, int n2, T eps, T* M_){
    if(n1*n2==0) return;
    int m = n2;
    int n = n1;
    int k = (m<n?m:n);
    T* tU =mem::aligned_new<T>(m*k);
    T* tS =mem::aligned_new<T>(k);
    T* tVT=mem::aligned_new<T>(k*n);
    int INFO=0;
    char JOBU  = 'S';
    char JOBVT = 'S';
    int wssize = 3*(m<n?m:n)+(m>n?m:n);
    int wssize1 = 5*(m<n?m:n);
    wssize = (wssize>wssize1?wssize:wssize1);
    T* wsbuf = mem::aligned_new<T>(wssize);
    tsvd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k,
        wsbuf, &wssize, &INFO);
    if(INFO!=0)
      std::cout<<INFO<<'\n';
    assert(INFO==0);
    mem::aligned_delete<T>(wsbuf);
    T eps_=tS[0]*eps;
    for(int i=0;i<k;i++)
      if(tS[i]<eps_)
        tS[i]=0;
      else
        tS[i]=1.0/tS[i];
    for(int i=0;i<m;i++){
      for(int j=0;j<k;j++){
        tU[i+j*m]*=tS[j];
      }
    }
    tgemm<T>('T','T',n,m,k,1.0,&tVT[0],k,&tU[0],m,0.0,M_,n);
    mem::aligned_delete<T>(tU);
    mem::aligned_delete<T>(tS);
    mem::aligned_delete<T>(tVT);
  }

}//end namespace
}//end namespace

