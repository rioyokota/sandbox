#ifndef logger_h
#define logger_h
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sys/time.h>
#include "types.h"

Timer timer;                                                    //!< Timer base value

//! Timer function
double get_time() {
  struct timeval tv;                                            // Time value
  gettimeofday(&tv, NULL);                                      // Get time of day in seconds and microseconds
  return double(tv.tv_sec+tv.tv_usec*1e-6);                     // Combine seconds and microseconds and return
}

//! Start timer for given event
inline void startTimer(std::string event) {
  timer[event] = get_time();                                    // Get time of day and store in beginTimer
}

//! Stop timer for given event
void stopTimer(std::string event) {
  std::cout << std::setw(20) << std::left                       //  Set format
	    << event << " : " << std::setprecision(7) << std::fixed
	    << get_time()-timer[event] << " s" << std::endl;    //  Print event and timer
}

#endif
