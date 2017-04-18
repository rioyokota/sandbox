extern "C" void matrix(double ** x, int nx, int ny) {
  for (int j=0; j<ny; j++)
    for (int i=0; i<nx; i++)
      x[i][j] = i + 10 * j;
}
