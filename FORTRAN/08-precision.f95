program precision
    implicit none

    integer, parameter :: ikind=selected_real_kind(6)
    real :: x, y, z

    ! double precision :: x, y, z

    x = 10.0
    y = 3.0

    z = x / y
    print *, z
    
end program precision