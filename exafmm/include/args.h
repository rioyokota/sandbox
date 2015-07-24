#pragma once
#include <cstdio>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <iomanip>

static struct option long_options[] = {
  {"numBodies",    1, 0, 'n'},
  {"leafSize",     1, 0, 'l'},
  {"threads",      1, 0, 't'},
  {"images",       1, 0, 'i'},
  {"mac",          1, 0, 'm'},
  {"verbose",      1, 0, 'v'},
  {"distribution", 1, 0, 'd'},
  {"repeat",       1, 0, 'r'},
  {"help",         0, 0, 'h'},
  {0, 0, 0, 0}
};

class Args {
public:
  int numBodies;
  int leafSize;
  int nspawn;
  int threads;
  int images;
  double mac;
  int useRmax;
  int useRopt;
  int mutual;
  int graft;
  int verbose;
  const char * distribution;
  int repeat;

private:
  void usage(char * name) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "Long option (short option)     : Description (Default value)\n"
            " --numBodies (-n)              : Number of bodies (%d)\n"
            " --leafSize (-l)               : Number of bodies per leaf cell (%d)\n"
            " --threads (-t)                : Number of threads (%d)\n"
            " --images (-i)                 : Number of periodic image levels (%d)\n"
            " --mac (-m)                    : Multipole acceptance criterion (%f)\n"
	    " --verbose (-v) [0/1]          : Print information to screen (%d)\n"
            " --distribution (-d) [l/c/s/p] : lattice, cube, sphere, plummer (%s)\n"
            " --repeat (-r)                 : Number of iteration loops (%d)\n"
            " --help (-h)                   : Show this help document\n",
            name,
            numBodies,
            leafSize,
	    threads,
            images,
            mac,
	    verbose,
            distribution,
	    repeat);
  }

  const char * parse(const char * arg) {
    switch (arg[0]) {
    case 'l':
      return "lattice";
    case 'c':
      return "cube";
    case 's':
      return "sphere";
    case 'p':
      return "plummer";
    default:
      fprintf(stderr, "invalid distribution %s\n", arg);
      exit(0);
    }
    return "";
  }

public:
  Args(int argc=0, char ** argv=NULL) : numBodies(1000000), leafSize(16), threads(16), images(0),
					mac(.4), verbose(1), distribution("cube"), repeat(1) {
    while (1) {
      int option_index;
      int c = getopt_long(argc, argv, "n:l:t:i:m:v:d:r:h", long_options, &option_index);
      if (c == -1) break;
      switch (c) {
      case 'n':
        numBodies = atoi(optarg);
        break;
      case 'l':
        leafSize = atoi(optarg);
        break;
      case 't':
        threads = atoi(optarg);
        break;
      case 'i':
        images = atoi(optarg);
        break;
      case 'm':
        mac = atof(optarg);
        break;
      case 'v':
	verbose= atoi(optarg);
	break;
      case 'd':
        distribution = parse(optarg);
        break;
      case 'r':
        repeat = atoi(optarg);
        break;
      case 'h':
        usage(argv[0]);
        exit(0);
      default:
        usage(argv[0]);
        exit(0);
      }
    }
  }

  void print(int stringLength, int P) {
    if (verbose) {                                              // If verbose flag is true
      std::cout << std::setw(stringLength) << std::fixed << std::left// Set format
		<< "numBodies" << " : " << numBodies << std::endl // Print numBodies  
		<< std::setw(stringLength)                      //  Set format
		<< "P" << " : " << P << std::endl               //  Print P
		<< std::setw(stringLength)                      //  Set format
		<< "MAC" << " : " << mac << std::endl           //  Print MAC
		<< std::setw(stringLength)                      //  Set format
		<< "leafSize" << " : " << leafSize << std::endl //  Print leafSize
		<< std::setw(stringLength)                      //  Set format
		<< "threads" << " : " << threads << std::endl   //  Print threads
		<< std::setw(stringLength)                      //  Set format
		<< "images" << " : " << images << std::endl     //  Print images
		<< std::setw(stringLength)                      //  Set format
		<< "verbose" << " : " << verbose << std::endl   //  Print verbose
		<< std::setw(stringLength)                      //  Set format
		<< "distribution" << " : " << distribution << std::endl// Print distribution
		<< std::setw(stringLength)                      //  Set format
		<< "repeat" << " : " << repeat << std::endl;    //  Print distribution
    }                                                           // End if for verbose flag
  }
};
