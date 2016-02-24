#ifndef logger_h
#define logger_h
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sys/time.h>

//! Timer
class Logger {
 typedef std::map<std::string,double> Timer;                    //!< Map of timer event name to timed value

 private:
  Timer beginTimer;                                             //!< Timer base value
  Timer timer;                                                  //!< Timings of all events

//! Timer function
  double get_time() const {
    struct timeval tv;                                          // Time value
    gettimeofday(&tv, NULL);                                    // Get time of day in seconds and microseconds
    return double(tv.tv_sec+tv.tv_usec*1e-6);                   // Combine seconds and microseconds and return
  }

 public:
//! Constructor
  Logger() : beginTimer(), timer() {}                           // Initializing class variables (empty)

//! Print timings of a specific event
  inline void printTime(std::string event) {
    std::cout << std::setw(20) << std::left                     //  Set format
	      << event << " : " << std::setprecision(7) << std::fixed
	      << timer[event] << " s" << std::endl;             //  Print event and timer
  }

//! Start timer for given event
  inline void startTimer(std::string event) {
    beginTimer[event] = get_time();                             // Get time of day and store in beginTimer
  }

//! Stop timer for given event
  double stopTimer(std::string event) {
    double endTimer = get_time();                               // Get time of day and store in endTimer
    timer[event] += endTimer - beginTimer[event];               // Accumulate event time to timer
    printTime(event);                                           // Print event and timer to screen
    return endTimer - beginTimer[event];                        // Return the event time
  }

  //! Print relative L2 norm error
  void printError(double diff1, double norm1) {
    std::cout << std::setw(20) << std::left << std::scientific  //  Set format
	      << "Rel. L2 Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;// Print potential error
  }
};
#endif
