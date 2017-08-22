subroutine calc_force(nstep, natom, part_type, coord, velocity, for_dpd, box_length, Epot)
  use dpd_param, only : aij_list
  implicit none
  ! global
  integer nstep, natom
  double precision coord(natom,3), velocity(natom,3), box_length(3)
  integer part_type(natom)
  double precision for_dpd(natom,3)
  double precision Epot
  ! local
  integer ii
  double precision dr(3), dv(3)
  double precision f(3), half_length(3)
  double precision aij, aij_half_cutoff
  integer iatom1, iatom2, itype1, itype2
  double precision E_each

  call check_bounds(natom, coord, box_length)

  Epot = 0.0d0
  for_dpd     = 0.0d0
  half_length(:) = box_length(:)/2.0d0

  !$OMP  parallel do &
  !$OMP& default(none) private(ii, iatom1, iatom2, itype1, itype2, &
  !$OMP& dr, dv, f, E_each, aij, aij_half_cutoff) &
  !$OMP& shared(natom, velocity, coord, part_type) &
  !$OMP& firstprivate(nstep, aij_list, box_length, half_length) &
  !$OMP& reduction(+:for_dpd, Epot)
  do iatom1 = 1, natom-1
    itype1 = part_type(iatom1)
    do iatom2 = iatom1+1, natom
      itype2 = part_type(iatom2)
      ! calc parameter
      aij = aij_list(1, itype1, itype2)
      aij_half_cutoff = aij_list(2, itype1, itype2)
      ! calc force
      dr(1:3) = coord(iatom1,1:3) - coord(iatom2,1:3)
      do ii = 1, 3
        if( dr(ii) .gt. half_length(ii))then
          dr(ii) = dr(ii) - box_length(ii)
        elseif( dr(ii) .lt. -half_length(ii))then
          dr(ii) = dr(ii) + box_length(ii)
        endif
      enddo
      dv(1:3) = velocity(iatom1,1:3) - velocity(iatom2,1:3)
      ! must be " iatom1 < iatom2 ", for random number
      call calc_force_body_TEA(dr, dv, f, E_each, aij, aij_half_cutoff, nstep, iatom1, iatom2)
      for_dpd(iatom1,1:3) = for_dpd(iatom1,1:3) + f(1:3)
      for_dpd(iatom2,1:3) = for_dpd(iatom2,1:3) - f(1:3)
      ! calc potential energy
      Epot = Epot + E_each
    enddo
  enddo
end subroutine
    
subroutine calc_force_bond(natom, coord, force_bond, box_length, Ebond)
  use dpd_vars, only : num_bonds, bond_list
  ! global
  integer natom
  double precision coord(natom,3)
  double precision force_bond(natom, 3)
  double precision box_length(3)
  double precision Ebond
  ! local
  integer ii, now_bond, iatom1, iatom2
  double precision dr(3), dr0(3), f(3), length, length2, dx
  double precision exppart, expminusone, const_exp
  double precision half_length(3)
  
  half_length(1:3) = box_length(1:3)/2.0d0

  Ebond = 0.0
  force_bond = 0.0d0
  !$OMP parallel do &
  !$OMP& default(none) private(now_bond, ii, iatom1, iatom2, const_exp, f, &
  !$OMP& expminusone, exppart, dx, dr, dr0, length, length2) &
  !$OMP& shared(coord, bond_list, num_bonds) &
  !$OMP& firstprivate(box_length, half_length) &
  !$OMP& reduction(+:force_bond, Ebond)
  do now_bond = 1, num_bonds
    iatom1 = bond_list(now_bond)%part1
    iatom2 = bond_list(now_bond)%part2
    dr(1:3) = coord(iatom1,1:3) - coord(iatom2,1:3)
    do ii = 1, 3
      if( dr(ii) .gt. half_length(ii))then
        dr(ii) = dr(ii) - box_length(ii)
      elseif( dr(ii) .lt. -half_length(ii))then
        dr(ii) = dr(ii) + box_length(ii)
      endif
    enddo
    length2 = dot_product(dr,dr)
    length = sqrt(length2)
    dr0(1:3) = dr(1:3)/length
    dx = bond_list(now_bond)%length - length
    if( bond_list(now_bond)%bond_type .eq. 1 )then
      f(1:3) =  bond_list(now_bond)%const*dx*dr0
      Ebond = Ebond + 0.5*bond_list(now_bond)%const*dx*dx
    elseif( bond_list(now_bond)%bond_type .eq. 2)then
      !Morse bond
      !r=dr.length();
      !exppart = exp(-bconst*(r-r0));
      !expminusone = exppart - 1.0;
      !ftmp = -2*aconst*bconst*exppart*expminusone * dr/r;
      !ene = aconst*expminusone*expminusone;
      exppart = exp(-bond_list(now_bond)%const2*abs(dx))
      expminusone = exppart - 1.0
      const_exp = bond_list(now_bond)%const*expminusone
      f(1:3) =  2.0d0*const_exp*bond_list(now_bond)%const2*exppart*dr0
      !f(1:3) =  -2.0d0*const1*const2*exppart*expminusone*dr0
      Ebond = Ebond + const_exp*expminusone
    endif
    force_bond(iatom1,1:3) = force_bond(iatom1,1:3) + f(1:3)
    force_bond(iatom2,1:3) = force_bond(iatom2,1:3) - f(1:3)
  enddo
end subroutine

subroutine calc_force_body_TEA(dr, dv, f, E_each, aij, aij_half_cutoff, nstep, iatom1, iatom2)
  use dpd_param, only : sigma_kdt, gamma, cutoff, cutoff2
  implicit none
  ! global
  integer nstep, iatom1, iatom2
  double precision dr(3), dv(3), f(3), E_each
  double precision aij, aij_half_cutoff
  ! local
  double precision l2, l, dr0(3), ip, ll, fpair, fdpd, frand, zeta
  double precision , parameter :: pi_2 = 2.0d0*dacos(-1.0d0)
  ! var for TEA
  integer(4) val(2), key(4)
  double precision rand(2)

  ! kdt = sqrt(1.0/dt)
  ! sigma = sqrt(2.0*kbt*gamma)
  ! dr = r0 - r1
  ! dv = v0 - v1
  ! l = sqrt(sum(dr*dr))
  ! dr0 = dr/l
  ! ip = dr0*dv
  ! ll = 1.0 - l/cutoff
  ! wd = ll*ll
  ! wr = abs(ll)
  ! zeta = gaussian_random_number
  ! f = aij*cutoff*ll*dr/l
  ! fdpd = -gamma*wd*ip*dr0
  ! frand = sigma*wr*zeta*kdt*dr0
  ! f = f + fdpd + frand
  ! ene = 0.5*aij*cutoff*ll*ll

  f = 0.0d0
  E_each = 0.0d0
  l2 = dot_product(dr,dr)
  if (l2 .ge. cutoff2) return
  l = sqrt(l2)
  dr0 = dr/l
  ip = dot_product(dr0,dv)
  ll = 1.0d0 - l/cutoff
  
  ! gaussian random number
  val(:) = 0
  key(1:2) = nstep + iatom1
  key(3:4) = nstep + iatom2
  call encrypt(val, key)
  rand(1:2) = abs(dble(val/2)/dble(2**30)) ! convert uint32_t to double precition 0 <= x < 1.0
  zeta = sqrt( -2.0*log(rand(1)))*sin(pi_2*rand(2))
  
  fpair = aij
  fdpd = -gamma*ll*ip
  frand = sigma_kdt*zeta
  f = (fpair + fdpd + frand)*ll*dr0
  E_each = aij_half_cutoff*ll*ll
  
end subroutine

