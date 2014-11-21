#pragma once
#include "logger.h"
#include "types.h"

namespace BoundBox {
  void bounds2box(Bounds & bounds) {
    bounds.X = (bounds.Xmax + bounds.Xmin) / 2;                 // Calculate center of domain
    bounds.R = std::max(max(bounds.X - bounds.Xmin),            // Calculate max distance from center
			max(bounds.Xmax - bounds.X));
    bounds.R *= 1.00001;                                        // Add some leeway to radius
    bounds.Xmin = bounds.X - bounds.R;                          // Update Xmin
    bounds.Xmax = bounds.X + bounds.R;                          // Update Xmax
  }

  Bounds getBounds(Bodies & bodies) {
    logger::startTimer("Get bounds");                           // Start timer
    Bounds bounds;                                              // Bounds : Contains Xmin, Xmax
    if (bodies.empty()) {                                       // If body vector is empty
      bounds.Xmin = bounds.Xmax = bounds.X = bounds.R = 0;      //  Set bounds to 0
    } else {                                                    // If body vector is not empty
      bounds.Xmin = bounds.Xmax = bodies.front().X;             //  Initialize Xmin, Xmax
      for (int b=0; b<int(bodies.size()); b++) {                //  Loop over bodies
        bounds.Xmin = min(bodies[b].X, bounds.Xmin);            //   Update Xmin
	bounds.Xmax = max(bodies[b].X, bounds.Xmax);            //   Update Xmax
      }                                                         //  End loop over bodies
    }                                                           // End if for empty body vector
    logger::stopTimer("Get bounds");                            // Stop timer
    return bounds;                                              // Return Xmin and Xmax
  }
};
