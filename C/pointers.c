#include <stdio.h>

void func1(int *aPtr);

int main()
{
	int a;
	printf("Please enter your number:");
	scanf("%d", &a);
	// printf("\n");
	
	printf("Your number is: %d\n", a);

	func1(&a);
	printf("Your number after calling func1: %d\n", a);
	printf("a = %d, is stored at %p\n", a, &a);

	return 0;
}

void func1(int *aPtr) // Call by Reference
{
	*aPtr *= 2;
	printf("Your number inside the func1 is: %d\n", *aPtr);
}