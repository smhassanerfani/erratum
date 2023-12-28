program namer
    ! This checks if the variables are defined correctly.
    implicit none

    ! Variables Declaration
    character :: name * 10

    ! Ask for writing the name
    print *, 'What is your name?'

    ! Read the name
    read *, name

    ! Print the name
    print *, 'My name is ', name

end program namer