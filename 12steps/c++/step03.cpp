extern "C" void minus(double * x, int n) {
  for (int i=0; i<n; i++)
    x[i] = -x[i];
}
