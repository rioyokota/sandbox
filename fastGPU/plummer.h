#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>

struct Plummer{
	std::vector<double> mass;
	std::vector<double3> pos, vel;
	Plummer(
			unsigned long n, 
			unsigned int  seed = 19810614, 
			const char   *filename = "plummer.dat") 
		: mass(n), pos(n), vel(n)
	{
		{
			std::ifstream ifs(filename);
			if(!ifs.fail()){
				unsigned long ntmp, stmp;
				ifs.read((char *)&ntmp, sizeof(unsigned long));
				ifs.read((char *)&stmp, sizeof(unsigned long));
				if(n == ntmp  && seed == stmp){
					ifs.read((char *)&mass[0], n*sizeof(double));
					ifs.read((char *)& pos[0], n*sizeof(double3));
					ifs.read((char *)& vel[0], n*sizeof(double3));
					return;
				}
			}
		}
		srand(seed);
		unsigned long i = 0;
		while(i < n){
			double X1 = my_rand();
			double X2 = my_rand();
			double X3 = my_rand();
			double R = 1.0/sqrt( (pow(X1,-2.0/3.0) - 1.0) );
			if(R < 100.0) {
				double Z = (1.0 - 2.0*X2)*R;
				double X = sqrt(R*R - Z*Z) * cos(2.0*M_PI*X3);
				double Y = sqrt(R*R - Z*Z) * sin(2.0*M_PI*X3);

				double Ve = sqrt(2.0)*pow( (1.0 + R*R), -0.25 );

				double X4 = 0.0; 
				double X5 = 0.0;

				while( 0.1*X5 >= X4*X4*pow( (1.0-X4*X4), 3.5) ) {
					X4 = my_rand(); X5 = my_rand(); 
				} 

				double V = Ve*X4;

				double X6 = my_rand();
				double X7 = my_rand();

				double Vz = (1.0 - 2.0*X6)*V;
				double Vx = sqrt(V*V - Vz*Vz) * cos(2.0*M_PI*X7);
				double Vy = sqrt(V*V - Vz*Vz) * sin(2.0*M_PI*X7);

				double conv = 3.0*M_PI/16.0;
				X *= conv; Y *= conv; Z *= conv;    
				Vx /= sqrt(conv); Vy /= sqrt(conv); Vz /= sqrt(conv);

				double M = 1.0;
				mass[i] = M/n;

				pos[i].x = X;
				pos[i].y = Y;
				pos[i].z = Z;

				vel[i].x = Vx;
				vel[i].y = Vy;
				vel[i].z = Vz;

				/*
				tmp_i = ldiv(i, 256);
				if(tmp_i.rem == 0) printf("i = %d \n", i);
				*/

				ldiv_t tmp_i = ldiv(i, n/64);

				if(tmp_i.rem == 0) {
					printf(".");
					fflush(stdout); 
				}
				i++; 
			}		
		} // while (i<n)
		double mcm = 0.0;

		double xcm[3], vcm[3];
		for(int k=0;k<3;k++) {
			xcm[k] = 0.0; vcm[k] = 0.0;
		} /* k */

		for(i=0; i<n; i++) {
			mcm += mass[i];
			xcm[0] += mass[i] * pos[i].x; 
			xcm[1] += mass[i] * pos[i].y; 
			xcm[2] += mass[i] * pos[i].z; 
			vcm[0] += mass[i] * vel[i].x; 
			vcm[1] += mass[i] * vel[i].y; 
			vcm[2] += mass[i] * vel[i].z; 
		}  /* i */
		for(int k=0;k<3;k++) {
			xcm[k] /= mcm; vcm[k] /= mcm;
		} /* k */

		for(i=0; i<n; i++) {
			pos[i].x -= xcm[0]; 
			pos[i].y -= xcm[1]; 
			pos[i].z -= xcm[2]; 
			vel[i].x -= vcm[0]; 
			vel[i].y -= vcm[1]; 
			vel[i].z -= vcm[2]; 
		} /* i */ 
		printf("\n");
		{
			std::ofstream ofs(filename);
			if(!ofs.fail()){
				unsigned long ntmp = n;
				unsigned long stmp = seed;
				ofs.write((char *)&ntmp, sizeof(unsigned long));
				ofs.write((char *)&stmp, sizeof(unsigned long));
				ofs.write((char *)&mass[0], n*sizeof(double));
				ofs.write((char *)& pos[0], n*sizeof(double3));
				ofs.write((char *)& vel[0], n*sizeof(double3));
			}
		}
	}
	static double my_rand(void) {
		return rand()/(1. + RAND_MAX);
	}
};
