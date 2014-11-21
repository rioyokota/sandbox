#pragma once
#include "logger.h"

//! Verify results
namespace Verify {
  //! Get sum of scalar component of a vector of fields
  double getSumScalar(Fields & fields, Bodies & bodies) {
    double v = 0;                                               // Initialize scalar
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields
      v += fields[f].p * bodies[f].q;                           //  Sum of scalar component
    }                                                           // End loop over fields
    return v;                                                   // Return scalar
  }

  //! Get norm of scalar component of a vector of fields
  double getNrmScalar(Fields & fields) {
    double v = 0;                                               // Initialize norm
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields
      v += fields[f].p * fields[f].p;                           //  Norm of scalar component
    }                                                           // End loop over fields
    return v;                                                   // Return norm
  }

  //! Get difference between scalar component of two vectors of fields
  double getDifScalar(Fields & fields, Fields & field2) {
    double v = 0;                                               // Initialize difference
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields & fields2
      v += (fields[f].p - fields2[f].p) * (fields[f].p - fields2[f].p); //  Difference of scalar component
    }                                                           // End loop over fields & fields2
    return v;                                                   // Return difference
  }

  //! Get difference between scalar component of two vectors of fields
  double getRelScalar(Fields & fields, Fields & fields2) {
    double v = 0;                                               // Initialize difference
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields & fields2
      v += ((fields[f].p - fields2[f].p) * (fields[f].p - fields2[f].p))
	/ (fields2[f].p * fields2[f].p);                        //  Difference of scalar component
    }                                                           // End loop over fields & fields2
    return v;                                                   // Return difference
  }

  //! Get norm of scalar component of a vector of fields
  double getNrmVector(Fields & fields) {
    double v = 0;                                               // Initialize norm
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields
      v += fields[f].X[0] * fields[f].X[0]                      //  Norm of vector x component
	+  fields[f].X[1] * fields[f].X[1]                      //  Norm of vector y component
	+  fields[f].X[2] * fields[f].X[2];                     //  Norm of vector z component
    }                                                           // End loop over fields
    return v;                                                   // Return norm
  }

  //! Get difference between scalar component of two vectors of fields
  double getDifVector(Fields & fields, Fields & fields2) {
    double v = 0;                                               // Initialize difference
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields & fields2
      v += (fields[f].X[0] - fields2[f].X[0]) * (fields[f].X[0] - fields2[f].X[0])  // Difference of vector x component
	+  (fields[f].X[1] - fields2[f].X[1]) * (fields[f].X[1] - fields2[f].X[1])  // Difference of vector y component
	+  (fields[f].X[2] - fields2[f].X[2]) * (fields[f].X[2] - fields2[f].X[2]); // Difference of vector z component
    }                                                           // End loop over fields & fields2
    return v;                                                   // Return difference
  }

  //! Get difference between scalar component of two vectors of fields
  double getRelVector(Fields & fields, Fields & fields2) {
    double v = 0;                                               // Initialize difference
    for (int f=0; f<int(fields.size()); f++) {                  // Loop over fields & fields2
      v += ((fields[f].X[0] - fields2[f].X[0]) * (fields[f].X[0] - fields2[f].X[0]) + // Difference of vector x component
	    (fields[f].X[1] - fields2[f].X[1]) * (fields[f].X[1] - fields2[f].X[1]) + // Difference of vector y component
	    (fields[f].X[2] - fields2[f].X[2]) * (fields[f].X[2] - fields2[f].X[2]))  // Difference of vector z component
	/ (fields2[f].X[0] * fields2[f].X[0] +                  //  Norm of vector x component
	   fields2[f].X[1] * fields2[f].X[1] +                  //  Norm of vector y component
	   fields2[f].X[2] * fields2[f].X[2]);                  //  Norm of vector z component
    }                                                           // End loop over fields & fields2
    return v;                                                   // Return difference
  }

  //! Print relative L2 norm scalar error
  void print(std::string title, double v) {
    if (logger::verbose) {                                      // If verbose flag is true
      std::cout << std::setw(logger::stringLength) << std::left //  Set format
		<< title << " : " << std::setprecision(logger::decimal) << std::scientific // Set title
                << v << std::endl;                              //  Print potential error
    }                                                           // End if for verbose flag
  }
};
