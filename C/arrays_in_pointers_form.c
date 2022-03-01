#include <stdio.h>

int main()
{
	int a[]={1, 2, 3, 4};

	printf("%p, %p \n", a, &a[0]);
	printf("%d, %d \n", *a, *&a[0]);
	
	printf("%d, %d \n", *(a+1), a[1]);

	return 0;

}