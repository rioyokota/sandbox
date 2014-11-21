#pragma once
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include "types.h"

class Dataset {                                                 // Contains all the different datasets
private:
  long filePosition;                                            // Position of file stream

private:
  //! Split range and return partial range
  void splitRange(int & begin, int & end, int iSplit, int numSplit) {
    assert(end > begin);                                        // Check that size > 0
    int size = end - begin;                                     // Size of range
    int increment = size / numSplit;                            // Increment of splitting
    int remainder = size % numSplit;                            // Remainder of splitting
    begin += iSplit * increment + std::min(iSplit,remainder);   // Increment the begin counter
    end = begin + increment;                                    // Increment the end counter
    if (remainder > iSplit) end++;                              // Adjust the end counter for remainder
  }

  //! Uniform distribution on [-1,1]^3 lattice
  Bodies lattice(int numBodies, int mpirank, int mpisize) {
    int nx = int(std::pow(numBodies*mpisize, 1./3));            // Number of points in x direction
    int ny = nx;                                                // Number of points in y direction
    int nz = nx;                                                // Number of points in z direction
    int begin = 0;                                              // Begin index in z direction
    int end = nz;                                               // End index in z direction
    splitRange(begin, end, mpirank, mpisize);                   // Split range in z direction
    int numLattice = nx * ny * (end - begin);                   // Total number of lattice points
    Bodies bodies(numLattice);                                  // Initialize bodies
    int b = 0;                                                  // Initialize body counter
    for (int ix=0; ix<nx; ix++) {                               // Loop over x direction
      for (int iy=0; iy<ny; iy++) {                             //  Loop over y direction
        for (int iz=begin; iz<end; iz++, b++) {                 //   Loop over z direction
          bodies[b].X[0] = (ix / real_t(nx-1)) * 2 - 1;         //    x coordinate
          bodies[b].X[1] = (iy / real_t(ny-1)) * 2 - 1;         //    y coordinate
          bodies[b].X[2] = (iz / real_t(nz-1)) * 2 - 1;         //    z coordinate
        }                                                       //   End loop over z direction
      }                                                         //  End loop over y direction
    }                                                           // End loop over x direction
    return bodies;                                              // Return bodies
  }

  //! Random distribution in [-1,1]^3 cube
  Bodies cube(int numBodies, int seed, int numSplit) {
    Bodies bodies(numBodies);                                   // Initialize bodies
    for (int i=0; i<numSplit; i++, seed++) {                    // Loop over partitions (if there are any)
      int begin = 0;                                            //  Begin index of bodies
      int end = bodies.size();                                  //  End index of bodies
      splitRange(begin, end, i, numSplit);                      //  Split range of bodies
      srand48(seed);                                            //  Set seed for random number generator
      for (int b=begin; b<end; b++) {                           //  Loop over bodies
	for (int d=0; d<3; d++) {                               //   Loop over dimension
	  bodies[b].X[d] = drand48() * 2 * M_PI - M_PI;         //    Initialize coordinates
	}                                                       //   End loop over dimension
      }                                                         //  End loop over bodies
    }                                                           // End loop over partitions
    return bodies;                                              // Return bodies
  }

  //! Random distribution on r = 1 sphere
  Bodies sphere(int numBodies, int seed, int numSplit) {
    Bodies bodies(numBodies);                                   // Initialize bodies
    for (int i=0; i<numSplit; i++, seed++) {                    // Loop over partitions (if there are any)
      int begin = 0;                                            //  Begin index of bodies
      int end = bodies.size();                                  //  End index of bodies
      splitRange(begin, end, i, numSplit);                      //  Split range of bodies
      srand48(seed);                                            //  Set seed for random number generator
      for (int b=begin; b<end; b++) {                           //  Loop over bodies
	for (int d=0; d<3; d++) {                               //   Loop over dimension
	  bodies[b].X[d] = drand48() * 2 - 1;                   //    Initialize coordinates
	}                                                       //   End loop over dimension
	real_t r = std::sqrt(norm(bodies[b].X));                //   Distance from center
	for (int d=0; d<3; d++) {                               //   Loop over dimension
	  bodies[b].X[d] /= r * 1.1;                            //    Normalize coordinates
	}                                                       //   End loop over dimension
      }                                                         //  End loop over bodies
    }                                                           // End loop over partitions
    return bodies;                                              // Return bodies
  }

  //! Plummer distribution in a r = M_PI/2 sphere
  Bodies plummer(int numBodies, int seed, int numSplit) {
    Bodies bodies(numBodies);                                   // Initialize bodies
    for (int i=0; i<numSplit; i++, seed++) {                    // Loop over partitions (if there are any)
      int begin = 0;                                            //  Begin index of bodies
      int end = bodies.size();                                  //  End index of bodies
      splitRange(begin, end, i, numSplit);                      //  Split range of bodies
      srand48(seed);                                            //  Set seed for random number generator
      int b=begin;                                              //  Body begin counter
      while (b < end) {                                         //  While body counter is within range
	real_t X1 = drand48();                                  //   First random number
	real_t X2 = drand48();                                  //   Second random number
	real_t X3 = drand48();                                  //   Third random number
	real_t R = 1.0 / sqrt( (pow(X1, -2.0 / 3.0) - 1.0) );   //   Radius
	if (R < 100.0) {                                        //   If radius is less than 100
	  real_t Z = (1.0 - 2.0 * X2) * R;                      //    z component
	  real_t X = sqrt(R * R - Z * Z) * cos(2.0 * M_PI * X3);//    x component
	  real_t Y = sqrt(R * R - Z * Z) * sin(2.0 * M_PI * X3);//    y component
	  real_t scale = 3.0 * M_PI / 16.0;                     //    Scaling factor
	  X *= scale; Y *= scale; Z *= scale;                   //    Scale coordinates
	  bodies[b].X[0] = X;                                   //    Assign x coordinate to body
	  bodies[b].X[1] = Y;                                   //    Assign y coordinate to body
	  bodies[b].X[2] = Z;                                   //    Assign z coordinate to body
	  b++;                                                  //    Increment body counter
	}                                                       //   End if for bodies within range
      }                                                         //  End while loop over bodies
    }                                                           // End loop over partitions
    return bodies;                                              // Return bodies
  }

public:
  //! Constructor
  Dataset() : filePosition(0) {}                                // Initialize variables

  //! Initialize source values
  void initSource(Bodies & bodies, int seed, int numSplit) {
    for (int i=0; i<numSplit; i++, seed++) {                    // Loop over partitions (if there are any)
      int begin = 0;                                            //  Begin index of bodies
      int end = bodies.size();                                  //  End index of bodies
      splitRange(begin, end, i, numSplit);                      //  Split range of bodies
      srand48(seed);                                            //  Set seed for random number generator
#if MASS
      for (int b=begin; b<end; b++) {                           //  Loop over bodies
	bodies[b].q = 1. / bodies.size();                       //   Initialize mass
      }                                                         //  End loop over bodies
#else
      real_t average = 0;                                       //  Initialize average charge
      for (int b=begin; b<end; b++) {                           //  Loop over bodies
	bodies[b].q = drand48() - .5;                           //   Initialize charge
	average += bodies[b].q;                                 //   Accumulate average
      }                                                         //  End loop over bodies
      average /= (end - begin);                                 //  Normalize average
      for (int b=begin; b<end; b++) {                           //  Loop over bodies
	bodies[b].q -= average;                                 //   Subtract average charge
      }                                                         //  End loop over bodies
#endif
    }                                                           // End loop over partitions
  }

  //! Initialize target values
  void initTarget(Bodies & bodies) {
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      bodies[b].p = 0;                                          //  Initialize potential
      bodies[b].X = 0;                                          //  Initialize force
    }                                                           // End loop over bodies
  }

  //! Initialize dsitribution, source & target value of bodies
  Bodies initBodies(int numBodies, const char * distribution,
		    int mpirank=0, int mpisize=1, int numSplit=1) {
    Bodies bodies;                                              // Initialize bodies
    switch (distribution[0]) {                                  // Switch between data distribution type
    case 'l':                                                   // Case for lattice
      bodies = lattice(numBodies,mpirank,mpisize);              //  Uniform distribution on [-1,1]^3 lattice
      break;                                                    // End case for lattice
    case 'c':                                                   // Case for cube
      bodies = cube(numBodies,mpirank,numSplit);                //  Random distribution in [-1,1]^3 cube
      break;                                                    // End case for cube
    case 's':                                                   // Case for sphere
      bodies = sphere(numBodies,mpirank,numSplit);              //  Random distribution on surface of r = 1 sphere
      break;                                                    // End case for sphere
    case 'p':                                                   // Case plummer
      bodies = plummer(numBodies,mpirank,numSplit);             //  Plummer distribution in a r = M_PI/2 sphere
      break;                                                    // End case for plummer
    default:                                                    // If none of the above
      fprintf(stderr, "Unknown data distribution %s\n", distribution);// Print error message
    }                                                           // End switch between data distribution type
    initSource(bodies,mpirank,numSplit);                        // Initialize source values
    initTarget(bodies);                                         // Initialize target values
    return bodies;                                              // Return bodies
  }

  //! Read source values from file
  void readSources(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "source" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ifstream file(name.str().c_str(),std::ios::in);        // Open file
    file.seekg(filePosition);                                   // Set position in file
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      file >> bodies[b].X[0];                                   //  Read data for x coordinates
      file >> bodies[b].X[1];                                   //  Read data for y coordinates
      file >> bodies[b].X[2];                                   //  Read data for z coordinates
      file >> bodies[b].q;                                      //  Read data for charge
    }                                                           // End loop over bodies
    filePosition = file.tellg();                                // Get position in file
    file.close();                                               // Close file
  }

  //! Write source values to file
  void writeSources(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "source" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ofstream file(name.str().c_str(),std::ios::out);       // Open file
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      file << bodies[b].X[0] << std::endl;                      //  Write data for x coordinates
      file << bodies[b].X[1] << std::endl;                      //  Write data for y coordinates
      file << bodies[b].X[2] << std::endl;                      //  Write data for z coordinates
      file << bodies[b].q  << std::endl;                        //  Write data for charge
    }                                                           // End loop over bodies
    file.close();                                               // Close file
  }

  //! Read target values from file
  void readTargets(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "target" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ifstream file(name.str().c_str(),std::ios::in);        // Open file
    file.seekg(filePosition);                                   // Set position in file
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      file >> bodies[b].p;                                      //  Read data for potential
      file >> bodies[b].X[0];                                   //  Read data for x acceleration
      file >> bodies[b].X[1];                                   //  Read data for y acceleration
      file >> bodies[b].X[2];                                   //  Read data for z acceleration
    }                                                           // End loop over bodies
    filePosition = file.tellg();                                // Get position in file
    file.close();                                               // Close file
  }

  //! Write target values to file
  void writeTargets(Bodies & bodies, int mpirank) {
    std::stringstream name;                                     // File name
    name << "target" << std::setfill('0') << std::setw(4)       // Set format
         << mpirank << ".dat";                                  // Create file name
    std::ofstream file(name.str().c_str(),std::ios::out);       // Open file
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      file << bodies[b].p << std::endl;                         //  Write data for potential
      file << bodies[b].X[0] << std::endl;                      //  Write data for x acceleration
      file << bodies[b].X[1] << std::endl;                      //  Write data for y acceleration
      file << bodies[b].X[2] << std::endl;                      //  Write data for z acceleration
    }                                                           // End loop over bodies
    file.close();                                               // Close file
  }

  //! Downsize target bodies by even sampling
  void sampleBodies(Bodies & bodies, int numTargets) {
    if (numTargets < int(bodies.size())) {                      // If target size is smaller than current
      int stride = bodies.size() / numTargets;                  //  Stride of sampling
      for (int i=0; i<numTargets; i++) {                        //  Loop over target samples
        bodies[i] = bodies[i*stride];                           //   Sample targets
      }                                                         //  End loop over target samples
      bodies.resize(numTargets);                                //  Resize bodies to target size
    }                                                           // End if for target size
  }

  //! Get bodies with positive charges
  Bodies getPositive(Bodies & bodies) {
    Bodies buffer = bodies;                                     // Copy bodies to buffer
    int b2 = 0;                                                 // Initialize counter of buffer
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      if (bodies[b].q >= 0) {                                   //  If source is positive
        buffer[b2] = bodies[b];                                 //   Copy data to buffer
        b2++;                                                   //   Increment counter
      }                                                         //  End if for positive source
    }                                                           // End loop over bodies
    buffer.resize(b2);                                          // Resize buffer
    return buffer;                                              // Return buffer
  }


  //! Get bodies with negative charges
  Bodies getNegative(Bodies & bodies) {
    Bodies buffer = bodies;                                     // Copy bodies to buffer
    int b2 = 0;                                                 // Initialize counter of buffer
    for (int b=0; b<int(bodies.size()); b++) {                  // Loop over bodies
      if (bodies[b].q < 0) {                                    //  If source is negative
        buffer[b2] = bodies[b];                                 //   Copy data to buffer
        b2++;                                                   //   Increment counter
      }                                                         //  End if for negative source
    }                                                           // End loop over bodies
    buffer.resize(b2);                                          // Resize buffer
    return buffer;                                              // Return buffer
  }
};
