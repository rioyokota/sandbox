subroutine VecRef( v, total)
  integer i, total, v(9)
  total = 0
  do i = 1,9
    total = total + v(i)
  end do
end
