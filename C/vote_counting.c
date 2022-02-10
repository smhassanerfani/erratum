#include <stdio.h>
#include <math.h>

int main()
{
	int i, j, k;
	int p[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0,
	   		1, 1, 2, 2, 2, 7, 7, 7, 7, 9,
			3, 4, 5, 6, 6, 4, 5, 5, 9, 8,
			0, 0, 0, 5, 5, 9, 9, 9, 8, 9, 
			0, 2, 2};

	int N = sizeof(p) / sizeof(p[0]);

	int f[10];
	for(i=0; i<10; i++)
	{
		f[i] = 0;
	}
	
	int sum = 0;
	float stdev = 0;
	float mean;

	for(j=0; j<N; j++)
	{
		f[p[j]]++;
		sum += p[j];
	}

	mean = (float)sum / N;

	for(j=0; j<N; j++)
	{
		stdev += pow((p[j] - mean), 2);
	}
	stdev = sqrt(stdev / (N-1));

	printf("the total is: %d\n", sum);
	printf("the average is: %.3f\n", mean);
	printf("the STD is: %.3f\n", stdev);

	printf("index\t freq.\t hist.\n");
	for(i=0; i<10; i++)
	{
		printf("%d\t %d\t ", i, f[i]);
		for(k=0; k<f[i]; k++)
		{
			printf("*");
		}
		printf("\n");
	}
	return 0;
}