AC_INIT([test], [1.0])
AC_CONFIG_SRCDIR([src/test.c])
AC_CONFIG_HEADER(config.h)
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_PROG_CC

AC_ARG_ENABLE(mpi, [AC_HELP_STRING([--enable-mpi],[compile FFTW MPI library])], enable_mpi=$enableval, enable_mpi=no)
if test "$enable_mpi" = "yes"; then
   if test $PRECISION = q; then
      AC_MSG_ERROR([quad precision is not supported in MPI])
   fi
   ACX_MPI([],[AC_MSG_ERROR([could not find mpi library for --enable-mpi])])
   AC_CHECK_PROG(MPIRUN, mpirun, mpirun)
   AC_SUBST(MPIRUN)

   save_CC=$CC
   CC=$MPICC
   AC_CHECK_SIZEOF(MPI_Fint, [], [#include <mpi.h>])
   CC=$save_CC
   if test 0 = $ac_cv_sizeof_MPI_Fint; then
      AC_MSG_WARN([sizeof(MPI_Fint) test failed]);
      AC_CHECK_SIZEOF(int)
      if test 0 = $ac_cv_sizeof_int; then AC_MSG_ERROR([sizeof(int) test failed]); fi
      ac_cv_sizeof_MPI_Fint=$ac_cv_sizeof_int
   fi
   C_MPI_FINT=C_INT`expr $ac_cv_sizeof_MPI_Fint \* 8`_T
   AC_SUBST(C_MPI_FINT)
fi
AM_CONDITIONAL(MPI, test "$enable_mpi" = "yes")

AX_CC_MAXOPT
AC_CONFIG_FILES([Makefile src/Makefile])
AC_OUTPUT
