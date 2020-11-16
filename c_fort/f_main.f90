program main
  integer :: v(9) = (/ 1,1,1,1,1,1,1,1,1 /)
  integer :: total = 0
  external VecSum
  call VecSum(v, total)
  print*,total
end program
