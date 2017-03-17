#ifndef timer_h
#define timer_h
#include <cstdio>
#include <map>
#include <string>
#include <sys/time.h>
#include "types.h"

namespace exafmm {
  //! Timer and Tracer logger
  namespace timer {
    typedef std::map<std::string,real_t> Timer;                 //!< Map of timer event name to timed value
    typedef Timer::iterator              T_iter;                //!< Iterator of timer event name map
    Timer beginTimer;                                           //!< Timer base value
    Timer timer;                                                //!< Timings of all events

    //! Timer function
    real_t get_time() {
      struct timeval tv;                                        // Time value
      gettimeofday(&tv, NULL);                                  // Get time of day in seconds and microseconds
      return real_t(tv.tv_sec)+real_t(tv.tv_usec)*1e-6;         // Combine seconds and microseconds and return
    }

    //! Start timer for given event
    inline void start(std::string event) {
      beginTimer[event] = get_time();                           // Get time of day and store in beginTimer
    }

    //! Stop timer for given event
    real_t stop(std::string event) {
      real_t endTimer = get_time();                             // Get time of day and store in endTimer
      timer[event] += endTimer - beginTimer[event];             // Accumulate event time to timer
      printf("%-20s : %f s\n", event.c_str(), timer[event]);    // Print time
      return endTimer - beginTimer[event];                      // Return the event time
    }
  };
}
#endif
