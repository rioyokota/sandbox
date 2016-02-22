#ifndef logger_h
#define logger_h
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <pthread.h>
#include <queue>
#include <string>
#include <sstream>
#include <sys/time.h>
#include <vector>

//! Timer and Trace logger
class Logger {
 typedef std::map<std::string,double>           Timer;          //!< Map of timer event name to timed value
 typedef std::map<std::string,double>::iterator T_iter;         //!< Iterator of timer event name map

 private:
  Timer beginTimer;                                             //!< Timer base value
  Timer timer;                                                  //!< Timings of all events

 public:
  int stringLength;                                             //!< Max length of event name
  int decimal;                                                  //!< Decimal precision
  bool verbose;                                                 //!< Print to screen

 private:
//! Timer function
  double get_time() const {
    struct timeval tv;                                          // Time value
    gettimeofday(&tv, NULL);                                    // Get time of day in seconds and microseconds
    return double(tv.tv_sec+tv.tv_usec*1e-6);                   // Combine seconds and microseconds and return
  }

 public:
//! Constructor
  Logger() : beginTimer(), timer(),                             // Initializing class variables (empty)
	     stringLength(20),                                  // Max length of event name
	     decimal(7),                                        // Decimal precision
	     verbose(false) {}                                  // Don't print timings by default

//! Print message to standard output
  inline void printTitle(std::string title) {
    if (verbose) {                                              // If verbose flag is true
      title += " ";                                             //  Append space to end of title
      std::cout << "--- " << std::setw(stringLength)            //  Align string length
                << std::left                                    //  Left shift
                << std::setfill('-')                            //  Set to fill with '-'
                << title << std::setw(10) << "-"                //  Fill until end of line
                << std::setfill(' ') << std::endl;              //  Set back to fill with ' '
    }                                                           // End if for verbose flag
  }

//! Start timer for given event
  inline void startTimer(std::string event) {
    beginTimer[event] = get_time();                             // Get time of day and store in beginTimer
  }

//! Stop timer for given event
  double stopTimer(std::string event) {
    double endTimer = get_time();                               // Get time of day and store in endTimer
    timer[event] += endTimer - beginTimer[event];               // Accumulate event time to timer
    if (verbose) printTime(event);                              // Print event and timer to screen
    return endTimer - beginTimer[event];                        // Return the event time
  }

//! Print timings of a specific event
  inline void printTime(std::string event) {
    if (verbose) {                                              // If verbose flag is true
      std::cout << std::setw(stringLength) << std::left         //  Set format
        << event << " : " << std::setprecision(decimal) << std::fixed
        << timer[event] << " s" << std::endl;                   //  Print event and timer
    }                                                           // End if for verbose flag
  }

//! Erase all events in timer
  inline void resetTimer() {
    timer.clear();                                              // Clear timer
  }

  //! Print relative L2 norm error
  void printError(double diff1, double norm1) {
    if (verbose) {                                              // If verbose flag is true
      std::cout << std::setw(stringLength) << std::left << std::scientific//  Set format
                << "Rel. L2 Error (pot)" << " : " << std::sqrt(diff1/norm1) << std::endl;// Print potential error
    }                                                           // End if for verbose flag
  }
};
#endif
