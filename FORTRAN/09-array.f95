PROGRAM Array
    IMPLICIT NONE

    ! Variable Declaration
    ! integer, parameter :: ikind = 5
    REAL, ALLOCATABLE, DIMENSION(:) :: x
    INTEGER :: elements
    
    elements = 5
    ALLOCATE(x(elements))

    x(1) = 1.00
    x(2) = 2.00
    x(3) = 3.00
    x(4) = 4.00
    x(5) = 5.00

    PRINT *, x

    DEALLOCATE(x)
    
END PROGRAM Array
