#include <stdio.h>
#define N 20

int main()
{
	int F[N];

	// initial conditions
	F[0] = 0;
	F[1] = 1;
	printf("F[1] = 1\n");
	int i;
	for(i = 2; i < N; i++)
	{
		F[i] = F[i-1] + F[i-2];
		printf("F[%d] = %d\n", i, F[i]);
	}
	return 0;
}