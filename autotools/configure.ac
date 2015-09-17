AC_INIT([test], [1.0])
AC_CONFIG_SRCDIR([src/test.c])
AC_CONFIG_AUX_DIR([build-aux])
AM_INIT_AUTOMAKE([foreign -Wall -Werror])
AC_PROG_CC([mpicc gcc])
AX_CFLAGS_WARN_ALL
AC_SUBST(CFLAGS)
AC_CONFIG_FILES([Makefile src/Makefile])
AC_CANONICAL_HOST
case $host_os in
  darwin*)
  ;;
  *)
  ;;
esac
AC_OUTPUT
