PROGRAM Precision
    IMPLICIT NONE

    INTEGER, PARAMETER :: IKIND=SELECTED_REAL_KIND(6)
    real :: x, y, z

    ! DOUBLE PRECISION :: x, y, z

    x = 10.0
    y = 3.0

    z = x / y
    PRINT *, z
    
END PROGRAM Precision
