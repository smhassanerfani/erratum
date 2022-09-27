#include <stdio.h>
#define TRUE -1
#define FALSE 0

int main()
{
	int n = 10;
	int a, s, i;

	i = 0;
	s = 0;
	while(TRUE) // Use i<n for limited entries
	{
		printf("Enter a number: ");
		scanf("%d", &a);
		if(a == -1) break;

		s += a;
		i++;
	}
	printf("\nSum of numbers: %d\n", s);
	
	float m;
	m = (float)s / i;
	printf("Average of numbers: %.2f\n", m);

	return 0;
}