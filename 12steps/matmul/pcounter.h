#include <cstring>
#include <papi.h>
#include <vector>

int                    PAPIEventSet = PAPI_NULL;            //!< PAPI event set
std::vector<char*>     PAPIEventNames;                      //!< Vector of PAPI event names
std::vector<int>       PAPIEventCodes;                      //!< Vector of PAPI event codes
std::vector<long long> PAPIEventValues;                     //!< Vector of PAPI event values

//! Start PAPI event
inline void startPAPI() {
  PAPI_library_init(PAPI_VER_CURRENT);                      // Initialize PAPI library
  char * allEvents = getenv("PAPI_EVENTS");                 // Get all PAPI event strings
  char eventName[256];                                      // PAPI event name
  while (allEvents) {                                       // While event string is not empty
    char * event = strchr(allEvents, ',');                  //  Get single event string
    int n = (event == NULL ? (int)strlen(allEvents) : event - allEvents);// Count string length
    int eventCode;                                          //  PAPI event code
    snprintf(eventName, n+1, "%s", allEvents);              //  Get PAPI event name
    if (PAPI_event_name_to_code(eventName, &eventCode) == PAPI_OK) {// Event name to event code
      PAPIEventNames.push_back(strdup(eventName));          //   Push event name to vector
      PAPIEventCodes.push_back(eventCode);                  //   Push event code to vector
    }                                                       //  End if for event name to event code
    if (event == NULL) break;                               //  Stop if event string is empty
    else allEvents = event + 1;                             //  Else move to next event string
  };                                                        // End while loop for event string
  if (!PAPIEventCodes.empty()) {                            // If PAPI events are set
    PAPI_create_eventset(&PAPIEventSet);                    // Create PAPI event set
    for (size_t i=0; i<PAPIEventCodes.size(); i++) {        // Loop over PAPI events
      PAPI_add_event(PAPIEventSet, PAPIEventCodes[i]);      //  Add PAPI event
    }                                                       // End loop over PAPI events
    PAPI_start(PAPIEventSet);                               // Start PAPI counter
  }
}

//! Stop PAPI event
inline void stopPAPI() {
  if (!PAPIEventCodes.empty()) {                            // If PAPI events are set
    PAPIEventValues.resize(PAPIEventCodes.size());          //  Resize PAPI event value vector
    PAPI_stop(PAPIEventSet, &PAPIEventValues[0]);           //  Stop PAPI counter
  }                                                         // End if for PAPI events
}

//! Print PAPI event
inline void printPAPI() {
  if (!PAPIEventCodes.empty()) {                            // If PAPI events are set and verbose is true
    printf("--- %-16s ------------\n", "PAPI events");      //  Print title
    for (size_t i=0; i<PAPIEventCodes.size(); i++) {        //  Loop over PAPI events
      printf("%-20s : %ld\n", PAPIEventNames[i], PAPIEventValues[i]); // Print PAPI event values
    }                                                       //  End loop over PAPI events
  }                                                         // End if for PAPI events
}
