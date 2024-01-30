PROGRAM Conditions

    IMPLICIT NONE

    ! Variables Declaration
    REAL :: x, y, ans
    INTEGER :: choice

    x = 12.0
    y = 3.0

    PRINT *, 'Please choose one option: '
    PRINT *, '1- Summation'
    PRINT *, '2- Multiplication'
    PRINT *, '3- Division'

    READ *, choice

    IF (choice == 1) THEN
        ans = x + y
    END IF
    IF (choice == 2) THEN
        ans = x * y
    END IF
    IF (choice == 3) THEN
        ans = x / y
    END IF

    PRINT *, 'According to what you chose, answer is: ', ans
END PROGRAM Conditions
