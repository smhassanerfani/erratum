program conditions

    implicit none

    ! Variables Declaration
    real :: x, y, ans
    integer :: choice

    x = 12.0
    y = 3.0

    print *, 'Please choose one option: '
    print *, '1- Summation'
    print *, '2- Multiplication'
    print *, '3- Division'

    read *, choice

    if (choice == 1) then
        ans = x + y
    end if
    if (choice == 2) then
        ans = x * y
    end if
    if (choice == 3) then
        ans = x / y
    end if

    print *, 'According to what you chose, answer is: ', ans
end program conditions