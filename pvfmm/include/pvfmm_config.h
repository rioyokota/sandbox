#ifndef _INCLUDE_PVFMM_CONFIG_H
#define _INCLUDE_PVFMM_CONFIG_H 1
 
/* include/pvfmm_config.h. Generated automatically at end of configure. */
/* config.h.  Generated from config.h.in by configure.  */
/* config.h.in.  Generated from configure.ac by autoheader.  */

/* Define to dummy `main' function (if any) required to link to the Fortran
   libraries. */
/* #undef F77_DUMMY_MAIN */

/* Define if F77 and FC dummy `main' functions are identical. */
/* #undef FC_DUMMY_MAIN_EQ_F77 */

/* Define if you have a BLAS library. */
/* #undef HAVE_BLAS */

/* Define to 1 if you have the <cstddef> header file. */
#ifndef PVFMM_HAVE_CSTDDEF
#define PVFMM_HAVE_CSTDDEF 1
#endif

/* Define to 1 if you have the <cstdlib> header file. */
#ifndef PVFMM_HAVE_CSTDLIB
#define PVFMM_HAVE_CSTDLIB 1
#endif


/* Define to 1 if you have the <dlfcn.h> header file. */
#ifndef PVFMM_HAVE_DLFCN_H
#define PVFMM_HAVE_DLFCN_H 1
#endif

/* Define if we have FFTW */
#ifndef PVFMM_HAVE_FFTW
#define PVFMM_HAVE_FFTW 1
#endif

/* Define if we have FFTW */
#ifndef PVFMM_HAVE_FFTWF
#define PVFMM_HAVE_FFTWF 1
#endif

/* Define to 1 if you have the `floor' function. */
#ifndef PVFMM_HAVE_FLOOR
#define PVFMM_HAVE_FLOOR 1
#endif

/* Define to 1 if you have the <inttypes.h> header file. */
#ifndef PVFMM_HAVE_INTTYPES_H
#define PVFMM_HAVE_INTTYPES_H 1
#endif

/* Define to 1 if you have the `dl' library (-ldl). */
#ifndef PVFMM_HAVE_LIBDL
#define PVFMM_HAVE_LIBDL 1
#endif

/* Define to 1 if you have the `imf' library (-limf). */
#ifndef PVFMM_HAVE_LIBIMF
#define PVFMM_HAVE_LIBIMF 1
#endif

/* Define to 1 if you have the `m' library (-lm). */
#ifndef PVFMM_HAVE_LIBM
#define PVFMM_HAVE_LIBM 1
#endif

/* Define to 1 if you have the `stdc++' library (-lstdc++). */
#ifndef PVFMM_HAVE_LIBSTDC__
#define PVFMM_HAVE_LIBSTDC__ 1
#endif

/* Define to 1 if your system has a GNU libc compatible `malloc' function, and
   to 0 otherwise. */
#ifndef PVFMM_HAVE_MALLOC
#define PVFMM_HAVE_MALLOC 1
#endif

/* Define to 1 if you have the <memory.h> header file. */
#ifndef PVFMM_HAVE_MEMORY_H
#define PVFMM_HAVE_MEMORY_H 1
#endif

/* Define to 1 if you have the `memset' function. */
#ifndef PVFMM_HAVE_MEMSET
#define PVFMM_HAVE_MEMSET 1
#endif

/* Define if compiler supports OpenMP */
#ifndef PVFMM_HAVE_OPENMP
#define PVFMM_HAVE_OPENMP 1
#endif

/* Define to 1 if you have the `pow' function. */
#ifndef PVFMM_HAVE_POW
#define PVFMM_HAVE_POW 1
#endif

/* Define if compiler supports quadruple precision */
#ifndef PVFMM_HAVE_QUAD_PRECISON
#define PVFMM_HAVE_QUAD_PRECISON 1
#endif

/* Define to 1 if you have the `sqrt' function. */
#ifndef PVFMM_HAVE_SQRT
#define PVFMM_HAVE_SQRT 1
#endif

/* Define to 1 if stdbool.h conforms to C99. */
#ifndef PVFMM_HAVE_STDBOOL_H
#define PVFMM_HAVE_STDBOOL_H 1
#endif

/* Define to 1 if you have the <stdint.h> header file. */
#ifndef PVFMM_HAVE_STDINT_H
#define PVFMM_HAVE_STDINT_H 1
#endif

/* Define to 1 if you have the <stdlib.h> header file. */
#ifndef PVFMM_HAVE_STDLIB_H
#define PVFMM_HAVE_STDLIB_H 1
#endif

/* Define to 1 if you have the <strings.h> header file. */
#ifndef PVFMM_HAVE_STRINGS_H
#define PVFMM_HAVE_STRINGS_H 1
#endif

/* Define to 1 if you have the <string.h> header file. */
#ifndef PVFMM_HAVE_STRING_H
#define PVFMM_HAVE_STRING_H 1
#endif

/* Define to 1 if you have the `strtol' function. */
#ifndef PVFMM_HAVE_STRTOL
#define PVFMM_HAVE_STRTOL 1
#endif

/* Define to 1 if you have the `strtoul' function. */
#ifndef PVFMM_HAVE_STRTOUL
#define PVFMM_HAVE_STRTOUL 1
#endif

/* Define to 1 if you have the <sys/stat.h> header file. */
#ifndef PVFMM_HAVE_SYS_STAT_H
#define PVFMM_HAVE_SYS_STAT_H 1
#endif

/* Define to 1 if you have the <sys/types.h> header file. */
#ifndef PVFMM_HAVE_SYS_TYPES_H
#define PVFMM_HAVE_SYS_TYPES_H 1
#endif

/* Define to 1 if you have the <unistd.h> header file. */
#ifndef PVFMM_HAVE_UNISTD_H
#define PVFMM_HAVE_UNISTD_H 1
#endif

/* Define to the sub-directory in which libtool stores uninstalled libraries. */
#ifndef PVFMM_LT_OBJDIR
#define PVFMM_LT_OBJDIR ".libs/"
#endif

/* Define to the address where bug reports for this package should be sent. */
#ifndef PVFMM_PACKAGE_BUGREPORT
#define PVFMM_PACKAGE_BUGREPORT "dmalhotra@ices.utexas.edu"
#endif

/* Define to the full name of this package. */
#ifndef PVFMM_PACKAGE_NAME
#define PVFMM_PACKAGE_NAME "PvFMM"
#endif

/* Define to the full name and version of this package. */
#ifndef PVFMM_PACKAGE_STRING
#define PVFMM_PACKAGE_STRING "PvFMM 1.0.0"
#endif

/* Define to the one symbol short name of this package. */
#ifndef PVFMM_PACKAGE_TARNAME
#define PVFMM_PACKAGE_TARNAME "pvfmm"
#endif

/* Define to the home page for this package. */
#ifndef PVFMM_PACKAGE_URL
#define PVFMM_PACKAGE_URL ""
#endif

/* Define to the version of this package. */
#ifndef PVFMM_PACKAGE_VERSION
#define PVFMM_PACKAGE_VERSION "1.0.0"
#endif

/* Path for precomputed data files. */
#ifndef PVFMM_PRECOMP_DATA_PATH
#define PVFMM_PRECOMP_DATA_PATH ""
#endif

/* Define if compiler supports quadruple precision */
//#ifndef PVFMM_QUAD_T
//#define PVFMM_QUAD_T __float128
//#endif

/* Define to 1 if you have the ANSI C header files. */
#ifndef PVFMM_STDC_HEADERS
#define PVFMM_STDC_HEADERS 1
#endif

#endif
