#ifndef args_h
#define args_h
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include "print.h"

namespace exafmm {
  static struct option long_options[] = {
    {"ncrit",        required_argument, 0, 'c'},
    {"distribution", required_argument, 0, 'd'},
    {"help",         no_argument,       0, 'h'},
    {"images",       required_argument, 0, 'i'},
    {"numBodies",    required_argument, 0, 'n'},
    {"path",         required_argument, 0, 'p'},
    {"P",            required_argument, 0, 'P'},
    {"theta",        required_argument, 0, 't'},
    {"verbose",      required_argument, 0, 'v'},
    {0, 0, 0, 0}
  };

  //! Parse and set the parameters of FMM and bodies from argv
  class Args {
  public:
    int ncrit;                                  //!< Number of bodies per leaf cell
    const char * distribution;                  //!< Body Distribution
    int images;                                 //!< Number of periodic images (3^images in each direction)
    int numBodies;                              //!< Number of bodies
    const char * path;                          //!< Path to save files
    int P;                                      //!< Order of expansions
    double theta;                               //!< Multipole acceptance criterion
    int verbose;                                //!< Verbose mode

  private:
    //! Print the usage of option-arguments
    void usage(char * name) {
      fprintf(stderr,
              "Usage: %s [options]\n"
              "Long option (short option)       : Description (Default value)\n"
              " --ncrit (-c)                    : Number of bodies per leaf cell (%d)\n"
              " --distribution (-d) [l/c/s/o/p] : lattice, cube, sphere, octant, plummer (%s)\n"
              " --help (-h)                     : Show this help document\n"
              " --images (-i)                   : Number of images (3^images in each direction) (%d)\n"
              " --numBodies (-n)                : Number of bodies (%d)\n"
              " --path (-p)                     : Path to save files (%s)\n"
              " --P (-P)                        : Order of expansion (%d)\n"
              " --theta (-t)                    : Multipole acceptance criterion (%f)\n"
              " --verbose (-v)                  : Print information to screen (%d)\n",
              name,
              ncrit,
              distribution,
              images,
              numBodies,
              path,
              P,
              theta,
              verbose);
    }

    //! Parse body distribution from option-argument (optarg)
    const char * parseDistribution(const char * arg) {
      switch (arg[0]) {
        case 'c': return "cube";
        case 'l': return "lattice";
        case 'o': return "octant";
        case 'p': return "plummer";
        case 's': return "sphere";
        default:
          fprintf(stderr, "invalid distribution %s\n", arg);
          abort();
      }
      return "";
    }

  public:
    //! Set default values to FMM parameters and parse argv for user-defined options
    Args(int argc=0, char ** argv=NULL)
      : ncrit(64),
        distribution("cube"),
        images(4),
        numBodies(10000),
        path("./"),
        P(10),
        theta(.4),
        verbose(1) {
      while (1) {
        int option_index;
        int c = getopt_long(argc, argv, "c:d:hi:n:p:P:t:v:",
                            long_options, &option_index);
        if (c == -1) break;
        switch (c) {
          case 'c':
            ncrit = atoi(optarg);
            break;
          case 'd':
            distribution = parseDistribution(optarg);
            break;
          case 'h':
            usage(argv[0]);
            abort();
          case 'i':
            images = atoi(optarg);
            break;
          case 'n':
            numBodies = atoi(optarg);
            break;
          case 'p':
            path = optarg;
            break;
          case 'P':
            P = atoi(optarg);
            break;
          case 't':
            theta = atof(optarg);
            break;
          case 'v':
            verbose = atoi(optarg);
            break;
          default:
            usage(argv[0]);
            abort();
        }
      }
      if (strcmp(distribution, "cube") != 0) {
        images = 0;
        printf("Setting images to 0 for distribution != cube\n");
      }
    }

    //! Print formatted output for arguments
    void show() {
      if (verbose) {
        print("ncrit", ncrit);
        print("distribution", distribution);
        print("images", images);
        print("numBodies", numBodies);
        print("path", path);
        print("P", P);
        print("theta", theta);
        print("verbose", verbose);
      }
    }
  };
}
#endif
