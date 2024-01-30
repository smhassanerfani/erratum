PROGRAM Table

    IMPLICIT NONE

    ! Variables Declaration
    REAL :: x, y, z

    OPEN(10, FILE='table.txt')
    WRITE(10, *) 'x          y          z'

    DO x=1, 5
        DO y=1, 10, 0.5
            z = x * y
            PRINT *, x, y, z
            WRITE(10, *) x, y, z
        END DO
    END DO

END PROGRAM Table
