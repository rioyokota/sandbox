#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>

class Dataset {
public:
  std::vector<kvec4> pos;
  Dataset(unsigned long n,
	  const char *filename = "plummer.dat") : pos(n) {
    std::fstream file;
    file.open(filename,std::ios::in);
    if (!file.fail()) {
      unsigned long ntmp;
      file.read((char *)&ntmp, sizeof(unsigned long));
      if (n == ntmp) {
	file.read((char *)&pos[0], n*sizeof(double4));
	return;
      }
    }
    file.close();
    unsigned long i = 0;
    while (i < n) {
      float X1 = drand48();
      float X2 = drand48();
      float X3 = drand48();
      float R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
      if (R < 100.0) {
	float Z = (1.0 - 2.0 * X2) * R;
	float X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * X3);
	float Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * X3);
	float conv = 3.0 * M_PI / 16.0;
	X *= conv; Y *= conv; Z *= conv;
	pos[i][0] = X;
	pos[i][1] = Y;
	pos[i][2] = Z;
	pos[i][3] = drand48() / n;
	ldiv_t tmp_i = ldiv(i, n/33);
	if(tmp_i.rem == 0) {
	  printf(".");
	  fflush(stdout);
	}
	i++;
      }
    }
    kvec4 com(0.0);
    for (i=0; i<n; i++) {
      com[0] += abs(pos[i][3]) * pos[i][0];
      com[1] += abs(pos[i][3]) * pos[i][1];
      com[2] += abs(pos[i][3]) * pos[i][2];
      com[3] += abs(pos[i][3]);
    }
    com[0] /= com[3];
    com[1] /= com[3];
    com[2] /= com[3];

    for(i=0; i<n; i++) {
      pos[i][0] -= com[0];
      pos[i][1] -= com[1];
      pos[i][2] -= com[2];
    }
    printf("\n");
    file.open(filename,std::ios::out);
    file.write((char *)&n, sizeof(unsigned long));
    file.write((char *)&pos[0], n*sizeof(double4));
    file.close();
  }
};
