PROGRAM ReadData

    IMPLICIT NONE

    ! Variables Declaration
    REAL :: x, y, z, k

    OPEN(10, FILE='data.txt')
    READ(10, *) x, y, z, k

    PRINT *, x, y, z, k

END PROGRAM ReadData
