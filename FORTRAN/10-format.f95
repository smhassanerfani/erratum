PROGRAM Fortran
    
    IMPLICIT NONE

    ! Variable Declaration
    DOUBLE PRECISION, ALLOCATABLE, DIMENSION(:) :: mat
    INTEGER :: i, j

    PRINT *, 'Enter the number of Elements: '
    READ *, i

    ALLOCATE(mat(i))

    DO j = 1, i
        
        mat(j) = cos(0.1 * j)

    END  DO

    ! print *, mat
    WRITE(*, 1) mat
    1 FORMAT(2F10.5)

    DEALLOCATE(mat)

END PROGRAM
