subroutine output_xyz(nstep, natom, part_type, coord, velocity, box_length)
  use filenames
  implicit none
  ! global
  integer nstep, natom, part_type(natom)
  double precision coord(natom,3), velocity(natom,3), box_length(3)
  ! local
  integer ii
  double precision half_length(1:3)
  character(3) atom_list(20)
  character(85) xyz_file_step, istep

  atom_list(1)  = "H  "
  atom_list(2)  = "He "
  atom_list(3)  = "Li "
  atom_list(4)  = "Be "
  atom_list(5)  = "B  "
  atom_list(6)  = "C  "
  atom_list(7)  = "N  "
  atom_list(8)  = "O  "
  atom_list(9)  = "F  "
  atom_list(10) = "Ne "
  atom_list(11) = "Na "
  atom_list(12) = "Mg "
  atom_list(13) = "Al "
  atom_list(14) = "Si "
  atom_list(15) = "P  "
  atom_list(16) = "S  "
  atom_list(17) = "Cl "
  atom_list(18) = "Ar "
  atom_list(19) = "K  "
  atom_list(20) = "Ca "


  half_length = box_length / 2.0d0
  write(istep,'(i8.8)') nstep
  xyz_file_step = trim(adjustl(xyz_file))//'_'//trim(adjustl(istep))
  xyz_file_step = trim(adjustl(xyz_file_step))
  open(30,file=xyz_file_step)

  write(30,"(i15)") natom
  write(30,"(a30)") "# part type, x, y, z, vx, vy, vz"
  do ii = 1, natom
    write(30,"(a3,6f15.3)") atom_list(part_type(ii)) , coord(ii,1), coord(ii,2), coord(ii,3), &
      velocity(ii,1), velocity(ii,2), velocity(ii,3)
  enddo
  close(30)
end subroutine

subroutine read_position_velocity(filename,natom,coord)
  implicit none
  ! global
  character(80) filename
  integer natom
  double precision coord(natom,3)
  ! local
  double precision dtemp(3)
  integer numter, iatom

  open(20,file=filename)

  do numter=1,natom
    read(20,*) iatom ,dtemp(1), dtemp(2), dtemp(3)
    coord(iatom,1:3) = dtemp(1:3)
  end do
  close(20)
end subroutine

subroutine random_initial_vel(natom,velocity)
  implicit none
  ! global
  integer natom
  double precision velocity(natom,3)
  ! local
  integer ii, ij
  double precision drand

  do ii = 1, natom
    do ij = 1, 3
      call random_number(drand)
      velocity(ii,ij) = drand -0.5d0
    enddo
  enddo
end subroutine

subroutine output_energy_dat(dt, nstep, natom,ene)
  use energy_module
  use filenames, only : ene_file
  implicit none
  ! global
  double precision dt
  integer nstep, natom
  type ( energy_var ) ene
  ! local

  ene%pot       = ene%pot    / dble(natom)
  ene%kin       = ene%kin    / dble(natom)
  ene%potave    = ene%potave / dble(natom)
  ene%kinave    = ene%kinave / dble(natom)

  if( nstep .eq. 1 )then
    open(31, file=ene_file)
    write(31,"(a95)") "#    1      2     3          4          5        6         7      8            9           10"
    write(31,"(a95)") "# step,  time, total, pot_total, kin_total, total_ave, pot_ave, kin_ave, pot_nonbond, pot_bond"
  endif
  write(31,"(i8,9f18.8)") nstep, dt*dble(nstep), ene%pot(1)+ene%kin, ene%pot(1), ene%kin, &
    ene%potave(1)+ene%kinave, ene%potave(1), ene%kinave, ene%pot(2), ene%pot(3)

end subroutine

subroutine output_checkpoint(filename, natom, coord)
  implicit none
  ! global
  character(80) filename
  integer natom
  double precision coord(natom,3)
  ! local
  integer iatom

  open(32, file= filename)
  do iatom = 1, natom
    write(32,"(i8,3f18.5)") iatom, coord(iatom,1), coord(iatom,2), coord(iatom,3)
  enddo

  close(32)
end subroutine

subroutine read_parameter()
  use simulation_setting
  use dpd_param
  use cell_index_param
  use filenames
  implicit none
  ! local
  integer ii
  character (len=255) :: finp ! input file name

  namelist /input1/ natom, nstep_total, dump_step, dump_ene_step
  namelist /data_file/ init_pos_file, init_vel_file, aij_file, particle_file, bond_file, &
    pos_file, vel_file, xyz_file, ene_file, input_dump_file
  namelist /phys_param/ box_length, dt, kb, lambda, temperature, rand_seed
  namelist /calc_flags/ use_random_initial_coord
  namelist /dpd_parameter/ gamma, cutoff

  call get_command_argument(1, finp)
  open(99, file=finp, status='old')

  read(99, input1)

  read(99, data_file)
  
  read(99, phys_param )
  half_dt_mass = (dt/2.0d0)

  read(99, dpd_parameter)
  kdt = dsqrt(1.0d0/dt)
  sigma = dsqrt(2.0*kb*temperature*gamma)
  sigma_kdt = sigma*kdt
  cutoff2 = (cutoff)**2

  ! calc cell_index_param
  cell_size(1:3) = int(box_length(1:3)/(cutoff))
  do ii = 1, 3
    if( cell_size(ii) .le. 2)then
      write(*,*) "cell_size is too small. cell_size = ", cell_size(ii)
    endif
  enddo
  allocate( cell_counter(0:cell_size(1)+1,0:cell_size(2)+1,0:cell_size(3)+1))

  read(99,calc_flags)

  close(99)

  ! dump input data
  open(34, file=input_dump_file)
  call output_now_time(34, 'tot_start ')

  write(34,"(a30)")    "&input1"
  write(34,"(a30,i10)")"  natom = ", natom
  write(34,"(a30,i10)")"  nstep_total   = ", nstep_total
  write(34,"(a30,i10)")"  dump_step     = ", dump_step
  write(34,"(a30)")     "&end"
  write(34,"(a30)")""
  write(34,"(a30)")"&data_file"
  write(34,"(a30,a80)")" init_pos_file = " , init_pos_file
  write(34,"(a30,a80)")" init_vel_file = " , init_vel_file
  write(34,"(a30,a80)")" aij_file =      " , aij_file
  write(34,"(a30,a80)")" particle_file = " , particle_file
  write(34,"(a30,a80)")" bond_file     = " , bond_file
  write(34,"(a30,a80)")" pos_file      = " , pos_file
  write(34,"(a30,a80)")" vel_file      = " , vel_file
  write(34,"(a30,a80)")" xyz_file      = " , xyz_file
  write(34,"(a30,a80)")" ene_file      = " , ene_file
  write(34,"(a30,a80)")"input_dump_file= " , input_dump_file
  write(34,"(a30)")"&end"
  write(34,"(a30)")""
  write(34,"(a30)")"&phys_param"
  write(34,"(a30,f15.5)")"  box_length(1) = ", box_length(1)
  write(34,"(a30,f15.5)")"  box_length(2) = ", box_length(2)
  write(34,"(a30,f15.5)")"  box_length(3) = ", box_length(3)
  write(34,"(a30,f15.5)")"  dt          = ", dt
  write(34,"(a30,f15.5)")"  kb          = ", kb
  write(34,"(a30,f15.5)")"  temperature = ", temperature
  write(34,"(a30,f15.5)")"  lambda      = ", lambda
  write(34,"(a30,i15)")"  rand_seed   = ", rand_seed
  write(34,"(a30)")"&end"
  write(34,"(a30)")""
  write(34,"(a30)")"&dpd_parameter"
  write(34,"(a30,f15.5)")"  gamma         = ", gamma         
  write(34,"(a30,f15.5)")"  cutoff        = ", cutoff       
  write(34,"(a30)")"&end"
  write(34,"(a30,f15.5)")"  sigma         = ", sigma
  write(34,"(a30)")""
  write(34,"(a30)")"&calc_flags"
  write(34,"(a30,l3)")"  use_random_initial_coord   = ", use_random_initial_coord  
  write(34,"(a30)")"&end"

  write(34,"(a30,3i5)") 'cell_size = ', cell_size(1), cell_size(2), cell_size(3)

end subroutine

subroutine init_dpd_vars(natom)
  use simulation_setting, only : use_random_initial_coord, box_length, temperature
  use dpd_vars
  use filenames
  use dpd_param, only : aij_list
  implicit none
  ! global
  integer natom
  ! local
  integer part_num
  double precision now_temperature

  allocate(coord(natom,3), velocity(natom,3), force(natom,3))
  allocate(vel_dpd(natom,3), for_dpd(natom,3))
  allocate(force_bond(natom,3))
  allocate(part_type(natom), fix_flag(natom))

  coord = 0.0d0
  velocity = 0.0d0
  force = 0.0d0
  part_type = -1

  ! particle type reading
  call read_part_type(natom, part_type, fix_flag, particle_file)
  call read_part_max(aij_file, part_num)
  
  allocate(aij_list(2,part_num,part_num))
  call read_aij(aij_file, part_num, aij_list)
  ! bond reading
  call read_bond(bond_file)
  
  ! coord reading
  if( use_random_initial_coord )then
    call random_initial_coord(natom, coord, box_length)
    call random_initial_vel(natom, velocity)
    call clear_f_or_v_for_fixed(natom, velocity)
    
    ! relax structure
    call steepest_descent_method(natom, part_type, coord, box_length, 5)
  else
    call read_position_velocity(init_pos_file, natom,coord)
    call read_position_velocity(init_vel_file, natom,velocity)
  endif
  call check_bounds(natom, coord, box_length)
  
  ! init velocity
  call remove_flow( natom, velocity )
  now_temperature = ( sum(velocity(:,:)*velocity(:,:)))/( 3.0d0 * dble(sum(fix_flag)) )
  velocity(:,:) = dsqrt(temperature/now_temperature)*velocity(:,:)


end subroutine

subroutine read_bond(bond_file)
  use dpd_vars, only : bond_list, num_bonds
  implicit none
  ! global
  character(80) bond_file
  ! local
  integer bond_count, ii, part1, part2, itype
  double precision length, const, const2
  character(40) dummy
  
  bond_count = 0
  open(20, file = bond_file)
  do
    read(20,*, end=100) dummy
    bond_count = bond_count + 1
  enddo
  100 close(20)
  bond_count = bond_count - 1
  num_bonds = bond_count

  allocate(bond_list(num_bonds))

  open(20, file = bond_file)
  read(20,*) dummy
  do ii = 1, bond_count
  read(20,*) itype, part1, part2, length, const, const2
    bond_list(ii)%bond_type = itype
    bond_list(ii)%part1 = part1
    bond_list(ii)%part2 = part2
    bond_list(ii)%length = length
    bond_list(ii)%const = const
    if( bond_list(ii)%bond_type .eq. 1 )then
      bond_list(ii)%const2 = 0
    else
      bond_list(ii)%const2 = const2
    endif
  enddo
end subroutine

subroutine read_part_type(natom, part_type, fix_flag, particle_file)
  implicit none
  ! global
  integer natom, part_type(natom), fix_flag(natom)
  character(80) particle_file
  ! local
  integer ii, iatom, itemp1, itemp2
  character(40) dummy

  open(20, file = particle_file)
  read(20, *) dummy
  do ii = 1, natom
    read(20,*) iatom, itemp1, itemp2
    part_type(iatom) = itemp1
    fix_flag(iatom) = itemp2
  enddo
  close(20)
  ! check
  do ii = 1, natom
    if( part_type(ii) .eq. -1)then
      write(*,*) "not specified particle type. num = ", ii
    endif
  enddo

end subroutine

subroutine read_part_max(aij_file, part_num)
  implicit none
  ! global
  character(80) aij_file
  integer part_num
  ! local
  character(40) dummy
  integer i1, i2
  double precision dtemp
  
  part_num = 0
  open(20, file = aij_file)
  read(20,*) dummy
  do
    read(20,*, end=100) i1, i2, dtemp
    if( part_num .lt. i1 )then
      part_num = i1
    endif
    if( part_num .lt. i2 )then
      part_num = i2
    endif
  enddo
  100 close(20)
end subroutine

subroutine read_aij(aij_file, part_num, aij_list )
  use dpd_param, only : cutoff
  implicit none
  ! global
  character(80) aij_file
  integer part_num
  double precision aij_list(2, part_num, part_num)
  ! local
  character(40) dummy
  integer i1, i2, ii, ij
  double precision dtemp
  
  aij_list = -10000
  
  open(20, file = aij_file)
  read(20,*) dummy
  do
    read(20,*, end=100) i1, i2, dtemp
    aij_list(1, i1,i2) = dtemp
  enddo
  100 close(20)

  do ii = 1, part_num
    do ij = 1, part_num
      if( aij_list(1, ii,ij) .eq. -10000)then
        aij_list(1, ii,ij) = aij_list(1, ij,ii)
      endif
    enddo
  enddo
  
  do ii = 1, part_num
    do ij = 1, part_num
      if( aij_list(1, ii,ij) .eq. -10000)then
        write(*,*) 'aij parameter lack ', ii, ij
      endif
    enddo
  enddo

  ! calc aij_half_cutoff
  aij_list(2,:,:) = 0.5d0*cutoff*aij_list(1,:,:)
end subroutine

subroutine random_initial_coord(natom, coord, box_length)
  implicit none
  ! global
  integer natom
  double precision coord(natom,3), box_length(3)
  ! local
  integer iatom, ii
  double precision drand
  
  do iatom = 1, natom
    do ii = 1, 3
      call random_number(drand)
      coord(iatom,ii) = box_length(ii)*(drand -0.5d0)
    enddo
  enddo
end subroutine

subroutine output_now_time(fnum,mode)
  use time_measure
  implicit none
  !global
  integer fnum
  character(10) mode
  ! output time
  character (len=10) :: stime1, stime2, stime3  ! for output date
  ! local
  integer(8) ii
  integer(8) jst(8)
  integer(4) t_end, diff, t_max, t_rate
  double precision delta_time

  if( mode .eq. 'tot_start ')then
    call date_and_time(stime1,stime2,stime3,jst)
    write(fnum,"(3i6,3x,3(i2,a1),i3)") &
      (jst(ii),ii=1,3),jst(5),":",jst(6),":", jst(7),".",jst(8)
    call system_clock(t_start_total)
  elseif( mode .eq. 'tot_end   ')then
    call system_clock(t_end, t_rate, t_max )
    if ( t_end < t_start_total ) then
      diff = t_end - t_start_total + t_max
    else
      diff = t_end - t_start_total
    endif
    write(fnum,"(a,f15.3,a)") " elapsed time :", diff/dble(t_rate), " ( seconds )"
    write(fnum,"(a,f15.3,a)") " elapsed time :", diff/(dble(t_rate)*3600), " ( hours )"
  endif

  if( mode .eq. 'step_start')then
    call system_clock(t_start_per_step)
  elseif( mode .eq. 'step_end  ')then
    call system_clock(t_end, t_rate, t_max )
      diff = t_end - t_start_per_step + t_max
    if ( t_end < t_start_per_step ) then
      diff = t_end - t_start_per_step + t_max
    else
      diff = t_end - t_start_per_step
    endif
    delta_time = (dble(diff)/dble(t_rate*n_step_for_time))
    write(fnum,"(a,f15.9,a)") " elapsed time per step :", delta_time, " ( seconds )"
  endif
end subroutine

subroutine calc_init_bond(natom, coord, force_bond, box_length)
  use dpd_vars, only : num_bonds, bond_list
  ! global
  integer natom
  double precision coord(natom,3)
  double precision force_bond(natom, 3)
  double precision box_length(3)
  ! local
  integer ii, now_bond, iatom1, iatom2
  double precision dr(3), dr0(3), length
  double precision half_length(3)
  
  half_length(1:3) = box_length(1:3)/2.0d0

  force_bond = 0.0d0
  do now_bond = 1, num_bonds
    iatom1 = bond_list(now_bond)%part1
    iatom2 = bond_list(now_bond)%part2
    dr(1:3) = coord(iatom2,1:3) - coord(iatom1,1:3)
    do ii = 1, 3
      if( dr(ii) .gt. half_length(ii))then
        dr(ii) = dr(ii) - box_length(ii)
      elseif( dr(ii) .lt. -half_length(ii))then
        dr(ii) = dr(ii) + box_length(ii)
      endif
    enddo
    length = sqrt(sum(dr(1:3)*dr(1:3)))
    dr0(1:3) = bond_list(now_bond)%length*(dr(1:3)/length)
    coord(iatom2,1:3) = coord(iatom1,1:3) + dr0
  enddo
end subroutine

subroutine remove_flow(natom, velocity)
  implicit none
  ! global
  integer natom
  double precision velocity(natom, 3)
  ! local
  integer ii
  double precision c_mass_velocity(3)

  do ii = 1, 3
    c_mass_velocity(ii) = sum( velocity(1:natom,ii)) / dble(natom)
    velocity(:,ii) = velocity(:,ii) - c_mass_velocity(ii)
  enddo
end subroutine

subroutine check_bounds(natom, coord, box_length)
  implicit none
  ! global
  integer natom
  double precision coord(natom,3), box_length(3)
  ! local
  integer ii

  ! molecular moving
  do ii = 1, natom
    coord(ii,1:3) = coord(ii,1:3) - nint(coord(ii,1:3)/box_length(1:3))*box_length(1:3)
  enddo
end subroutine

! steepest descent method
! ref : http://www.sccj.net/CSSJ/jcs/v6n1/a1/document.pdf
subroutine steepest_descent_method(natom, part_type, coord, box_length, initialize_step)
  implicit none
  ! global
  integer natom, initialize_step
  double precision coord(natom,3)
  integer part_type(natom)
  double precision box_length(3)
  ! local
  double precision delta_E, dRMS, Epot
  double precision :: Epot_old = -1.0d100
  double precision :: delta_r = 1.0d0
  double precision , parameter :: delta_max = 20.0d0
  integer nstep
  double precision force_bond(natom,3), force(natom,3), Ebond
  double precision vel_dummy(natom, 3)

  vel_dummy = 0.0d0
  do nstep = 1, initialize_step
    call calc_init_bond(natom, coord, force_bond, box_length)
    call calc_force_bond(natom, coord, force_bond, box_length, Ebond)
    call calc_force(0, natom, part_type, coord, vel_dummy, force, box_length, Epot)
    force = force + force_bond
    call clear_f_or_v_for_fixed(natom, force)
    Epot = Ebond + Epot

    write(*,*) "#", nstep, Epot

    delta_E = Epot - Epot_old
    dRMS = dsqrt( sum( force(:,:)*force(:,:) ))

    ! converged ?
    if( (dRMS .lt. 0.001d0 ).or.(dabs(delta_E) .lt. 1.0d0))then
      if( Epot .lt. 0.0d0)then
        exit
      endif
    endif

    if( delta_E .gt. 0.0d0)then
      delta_r = 0.5*delta_r
    else
      delta_r = min( delta_max, 1.2*delta_r)
    endif

    coord = coord + (delta_r/dRMS)*force
    call check_bounds(natom, coord, box_length)
    Epot_old = Epot
  enddo
end subroutine

subroutine clear_f_or_v_for_fixed(natom, velocity)
  use dpd_vars, only : fix_flag
  implicit none
  ! global
  integer natom
  double precision velocity(natom,3)
  ! local
  integer ii

  do ii = 1, natom
    velocity(ii,1:3) = velocity(ii,1:3)*fix_flag(ii)
  enddo
end subroutine
