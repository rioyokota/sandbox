#pragma once
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>

class Dataset {
 public:
  std::vector<double4> pos;
  Dataset(unsigned long n, 
	  unsigned int seed = 19810614, 
	  const char *filename = "plummer.dat") : pos(n) {
    std::ifstream file(filename);
    if (!file.fail()) {
      unsigned long ntmp, stmp;
      file.read((char *)&ntmp, sizeof(unsigned long));
      file.read((char *)&stmp, sizeof(unsigned long));
      if (n == ntmp && seed == stmp) {
	file.read((char *)&pos[0], n*sizeof(double4));
	return;
      }
    }
    srand48(seed);
    unsigned long i = 0;
    while (i < n) {
      double X1 = drand48();
      double X2 = drand48();
      double X3 = drand48();
      double R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );
      if (R < 100.0) {
	double Z = (1.0 - 2.0 * X2) * R;
	double X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * X3);
	double Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * X3);
	double conv = 3.0 * M_PI / 16.0;
	X *= conv; Y *= conv; Z *= conv;
	pos[i].x = X;
	pos[i].y = Y;
	pos[i].z = Z;
	pos[i].w = drand48() / n;
	ldiv_t tmp_i = ldiv(i, n/64);
	if(tmp_i.rem == 0) {
	  printf(".");
	  fflush(stdout); 
	}
	i++; 
      }		
    }
    double4 com = {0.0};
    for (i=0; i<n; i++) {
      com.x += pos[i].w * pos[i].x; 
      com.y += pos[i].w * pos[i].y; 
      com.z += pos[i].w * pos[i].z; 
      com.w += pos[i].w;
    }
    com.x /= com.w;
    com.y /= com.w;
    com.z /= com.w;

    for(i=0; i<n; i++) {
      pos[i].x -= com.x; 
      pos[i].y -= com.y; 
      pos[i].z -= com.z; 
    }
    printf("\n");
    std::ofstream ofs(filename);
    if(!ofs.fail()){
      unsigned long ntmp = n;
      unsigned long stmp = seed;
      ofs.write((char *)&ntmp, sizeof(unsigned long));
      ofs.write((char *)&stmp, sizeof(unsigned long));
      ofs.write((char *)&pos[0], n*sizeof(double4));
    }
  }
};
