namespace pvfmm{

template <class T>
std::ostream& operator<<(std::ostream& output, const Permutation<T>& P){
  output<<std::setprecision(4)<<std::setiosflags(std::ios::left);
  size_t size=P.perm.Dim();
  for(size_t i=0;i<size;i++) output<<std::setw(10)<<P.perm[i]<<' ';
  output<<";\n";
  for(size_t i=0;i<size;i++) output<<std::setw(10)<<P.scal[i]<<' ';
  output<<";\n";
  return output;
}

template <class T>
Permutation<T>::Permutation(size_t size){
  perm.Resize(size);
  scal.Resize(size);
  for(size_t i=0;i<size;i++){
    perm[i]=i;
    scal[i]=1.0;
  }
}

template <class T>
Permutation<T> Permutation<T>::RandPerm(size_t size){
  Permutation<T> P(size);
  for(size_t i=0;i<size;i++){
    P.perm[i]=rand()%size;
    for(size_t j=0;j<i;j++)
      if(P.perm[i]==P.perm[j]){ i--; break; }
    P.scal[i]=((T)rand())/RAND_MAX;
  }
  return P;
}

template <class T>
Matrix<T> Permutation<T>::GetMatrix() const{
  size_t size=perm.Dim();
  Matrix<T> M_r(size,size,NULL);
  for(size_t i=0;i<size;i++)
    for(size_t j=0;j<size;j++)
      M_r[i][j]=(perm[j]==i?scal[j]:0.0);
  return M_r;
}

template <class T>
size_t Permutation<T>::Dim() const{
  return perm.Dim();
}

template <class T>
Permutation<T> Permutation<T>::Transpose(){
  size_t size=perm.Dim();
  Permutation<T> P_r(size);

  Vector<PERM_INT_T>& perm_r=P_r.perm;
  Vector<T>& scal_r=P_r.scal;
  for(size_t i=0;i<size;i++){
    perm_r[perm[i]]=i;
    scal_r[perm[i]]=scal[i];
  }
  return P_r;
}

template <class T>
Permutation<T> Permutation<T>::operator*(const Permutation<T>& P){
  size_t size=perm.Dim();
  assert(P.Dim()==size);

  Permutation<T> P_r(size);
  Vector<PERM_INT_T>& perm_r=P_r.perm;
  Vector<T>& scal_r=P_r.scal;
  for(size_t i=0;i<size;i++){
    perm_r[i]=perm[P.perm[i]];
    scal_r[i]=scal[P.perm[i]]*P.scal[i];
  }
  return P_r;
}

template <class T>
Matrix<T> Permutation<T>::operator*(const Matrix<T>& M){
  if(Dim()==0) return M;
  assert(M.Dim(0)==Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  Matrix<T> M_r(d0,d1,NULL);
  for(size_t i=0;i<d0;i++){
    const T s=scal[i];
    const T* M_=M[i];
    T* M_r_=M_r[perm[i]];
    for(size_t j=0;j<d1;j++)
      M_r_[j]=M_[j]*s;
  }
  return M_r;
}

template <class T>
Matrix<T> operator*(const Matrix<T>& M, const Permutation<T>& P){
  if(P.Dim()==0) return M;
  assert(M.Dim(1)==P.Dim());
  size_t d0=M.Dim(0);
  size_t d1=M.Dim(1);

  Matrix<T> M_r(d0,d1,NULL);
  for(size_t i=0;i<d0;i++){
    const PERM_INT_T* perm_=&(P.perm[0]);
    const T* scal_=&(P.scal[0]);
    const T* M_=M[i];
    T* M_r_=M_r[i];
    for(size_t j=0;j<d1;j++)
      M_r_[j]=M_[perm_[j]]*scal_[j];
  }
  return M_r;
}

}//end namespace
