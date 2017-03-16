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
      title += " ";                                             //  Append space to end of title
      std::cout << "--- " << std::setw(stringLength)            //  Align string length
                << std::left                                    //  Left shift
                << std::setfill('-')                            //  Set to fill with '-'
                << title << std::setw(10) << "-"                //  Fill until end of line
                << std::setfill(' ') << std::endl;              //  Set back to fill with ' '
    }

    //! Start timer for given event
    inline void start(std::string event) {
      beginTimer[event] = get_time();                           // Get time of day and store in beginTimer
    }

    //! Print timings of a specific event
    inline void printTime(std::string event) {
      std::cout << std::setw(stringLength) << std::left         //  Set format
                << event << " : " << std::setprecision(decimal) << std::fixed
                << timer[event] << " s" << std::endl;           //  Print event and timer
    }

    //! Stop timer for given event
    double stop(std::string event, int print=1) {
      double endTimer = get_time();                             // Get time of day and store in endTimer
      timer[event] += endTimer - beginTimer[event];             // Accumulate event time to timer
      if (print) printTime(event);                              // Print event and timer to screen
      return endTimer - beginTimer[event];                      // Return the event time
    }
  };
}
#endif
