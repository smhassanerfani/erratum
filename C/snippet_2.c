#include <stdio.h>

int main()
{
	int a1, a2;
	int b1, b2;

	a1 = 10;
	a2 = 10;

	b1 = a1++;
	b2 = ++a2;

	printf("b1 = %d\n", b1);
	printf("b2 = %d\n", b2);

	return 0;
}