#include <stdio.h>

int main()
{
	int x;
	printf("Enter an integer:\n");
	scanf("%d", &x);
	x = -x--;
	printf("%d\n", x);

	if(x % 2 == 0)
	{
		printf("%d in EVEN.\n", x);
	}
	else
	{
		printf("%d is ODD.\n", x);
	}

	if(x > 0)
	{
		printf("%d is positive.\n", x);
	}
	else if (x < 0)
	{
		printf("%d is negative.\n", x);
	}
	else
	{
		printf("%d is zero.\n", x);
	}

	return 0;
}