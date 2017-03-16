#ifndef timer_h
#define timer_h
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <stdint.h>
#include <string>
#include <sstream>
#include <sys/time.h>
#include <vector>

namespace exafmm {
  //! Timer and Tracer logger
  namespace timer {
    typedef std::map<std::string,double> Timer;                 //!< Map of timer event name to timed value
    typedef Timer::iterator              T_iter;                //!< Iterator of timer event name map
    Timer beginTimer;                                           //!< Timer base value
    Timer timer;                                                //!< Timings of all events
    int stringLength = 20;                                      //!< Max length of event name
    int decimal = 7;                                            //!< Decimal precision

    //! Timer function
    double get_time() {
      struct timeval tv;                                        // Time value
      gettimeofday(&tv, NULL);                                  // Get time of day in seconds and microseconds
      return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;         // Combine seconds and microseconds and return
    }

    //! Print message to standard output
    inline void printTitle(std::string title) {
      printf("--- %-16s ------------\n", title.c_str());
    }

    //! Start timer for given event
    inline void start(std::string event) {
      beginTimer[event] = get_time();                           // Get time of day and store in beginTimer
    }

    //! Stop timer for given event
    double stop(std::string event) {
      double endTimer = get_time();                             // Get time of day and store in endTimer
      timer[event] += endTimer - beginTimer[event];             // Accumulate event time to timer
      printf("%-20s : %lf s\n", event.c_str(), timer[event]);   // Print time
      return endTimer - beginTimer[event];                      // Return the event time
    }
  };
}
#endif
