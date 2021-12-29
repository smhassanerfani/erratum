#include <stdio.h>
#include <math.h>

int main()
{
	float a, d, p;
	int n;

	printf("Enter the first element (a): ");
	scanf("%f", &a);
	printf("\n");

	printf("Enter the step size (d): ");
	scanf("%f", &d);
	printf("\n");
	
	printf("Enter the length of series (n): ");
	scanf("%d", &n);
	printf("\n");

	printf("Enter the exponent power (p): ");
	scanf("%f", &p);
	printf("\n");

	float ai = 0;
	float s = 0;

	float bi = 0;
	float t = 0;
	
	printf("i\ta(i)\ts(i)\tb(i)\tt(i)\n");

	/*
	int i = 0;
	while(i<n)
	{
		ai = a + i * d;
		s += ai;

		printf("%d\t%.2f\t%.2f\n", i, ai, s);

		i++;
	}
	*/

	for(int i = 0; i < n; i++)
	{
		ai = a + i * d;
		s += ai;

		bi = pow(ai, p);
		t += bi;

		printf("%d\t%.2f\t%.2f\t%.2f\t%.2f\n", i, ai, s, bi, t);		
	}

	return 0;
}