program fortran
    
    implicit none

    ! Variable Declaration
    double precision, allocatable, dimension(:) :: mat
    integer :: i, j

    print *, 'Enter the number of Elements: '
    read *, i

    allocate(mat(i))

    do j = 1, i
        
        mat(j) = cos(0.1 * j)

    end  do

    ! print *, mat
    write(*, 1) mat
    1 format(2f10.5)

    deallocate(mat)

end program