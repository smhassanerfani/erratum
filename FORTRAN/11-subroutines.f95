PROGRAM VariableNamingExample
    ! This program calculates the difference between volume of 2 spheres.
    implicit none
  
    ! Declare variables using lowercase letters
    DOUBLE PRECISION :: rad1, rad2, vol1, vol2
    CHARACTER(LEN=1) :: response
  
    DO
        ! Initialize variables
        PRINT *, 'Please enter the radians of two spheres:'
        READ *, rad1, rad2
    
        ! Call a subroutine with mixed case
        CALL CalculateVolume(rad1, vol1)
        CALL CalculateVolume(rad2, vol2)

        WRITE(*, 10) 'The volume difference is: ', abs(vol1 - vol2)
        10 FORMAT(a, F10.4)
        
        PRINT *, 'Do you want to continue? (Y/y)'
        READ *, response
        IF (response /= 'Y' .AND. response /= 'y') STOP
    
    END DO
  
    
    CONTAINS
  
    ! Subroutine to print a summary
    SUBROUTINE CalculateVolume(r, v)
        REAL :: pi
        DOUBLE PRECISION, INTENT(IN) :: r
        DOUBLE PRECISION, INTENT(OUT) :: v

        pi = 4.0 * atan(1.0)
        
        ! Calculate the volume
        v = (4.0/3.0) * pi * (r**3) 

    END SUBROUTINE CalculateVolume
  
  END PROGRAM VariableNamingExample
  