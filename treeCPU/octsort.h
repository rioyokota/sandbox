template <int RS>
inline void octsort64(int np, Key_index *key_index, Key_index *buf)
{
  int count[64];
  int off[65];
  off[0] = 0;
  for(int k=0; k<64; k++){
    count[k] = 0;
  }
  for(int i=0; i<np; i++){
    int k = (key_index[i].key >> RS) & 63;
    count[k]++;
  }
  for(int k=0; k<64; k++){
    off[k+1] = off[k] + count[k];
  }
  for(int i=0; i<np; i++){
    int k = (key_index[i].key >> RS) & 63;
    buf[off[k]++] = key_index[i];
    __builtin_prefetch(&buf[0+off[k]]);
  }
  for(int k=0; k<64; k++){
    off[k] -= count[k];
  }
  for(int i=0; i<np; i++){
    key_index[i] = buf[i];
  }
  for(int k=0; k<64; k++){
    if(count[k] >= Node::NLEAF){
      octsort64 <RS-6> (count[k], key_index + off[k], buf);
    }
  }
}

template <>
void octsort64 <3> (int, Key_index*, Key_index*){
  assert(0);
}

template <>
void octsort64 <57> (int np, Key_index *key_index, Key_index *buf){
  const int RS = 57;
  int count[64];
  int off[65];
  off[0] = 0;
  for(int k=0; k<64; k++){
    count[k] = 0;
  }
  for(int i=0; i<np; i++){
    int k = (key_index[i].key >> RS) & 63;
    count[k]++;
  }
  for(int k=0; k<64; k++){
    off[k+1] = off[k] + count[k];
  }
  for(int i=0; i<np; i++){
    int k = (key_index[i].key >> RS) & 63;
    buf[off[k]++] = key_index[i];
    __builtin_prefetch(&buf[0+off[k]]);
  }
  for(int k=0; k<64; k++){
    off[k] -= count[k];
  }
  for(int i=0; i<np; i++){
    key_index[i] = buf[i];
  }
#pragma omp parallel for
  for(int k=0; k<64; k++){
    if(count[k] >= Node::NLEAF){
      octsort64 <RS-6> (count[k], key_index + off[k], buf + off[k]);
    }
  }
}
