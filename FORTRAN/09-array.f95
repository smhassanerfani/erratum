program array
    implicit none

    ! Variable Declaration
    ! integer, parameter :: ikind = 5
    real, allocatable, dimension(:) :: x
    integer :: elements
    
    elements = 5
    allocate(x(elements))

    x(1) = 1.00
    x(2) = 2.00
    x(3) = 3.00
    x(4) = 4.00
    x(5) = 5.00

    print *, x

    deallocate(x)
    
end program array