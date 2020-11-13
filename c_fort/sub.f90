module Sub
contains
  integer function VecSum(v)
    integer i, v(9)
    VecSum = 0
    do i = 1,9
      VecSum = VecSum + v(i)
    end do
  end
end module
