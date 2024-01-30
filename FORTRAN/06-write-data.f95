PROGRAM WriteData

    IMPLICIT NONE

    ! Variables Declaration
    REAL :: x, y, z

    x = 0.35
    y = 1.25
    z = 123.0

    OPEN(12, FILE='data-02.txt')
    WRITE(12, *) x, y, z

    PRINT *, 'Variables have been written in data-02.txt'

END PROGRAM WriteData
