#include <stdio.h>

void print_array(int lst[], int num);
float cal_mean(int lst[], int num);
void make_double(int lst[], int num);

int main()
{
	int p[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0};

	int n = sizeof(p) / sizeof(p[0]);

	print_array(p, n);
	
	printf("The mean value of vector p: %.3f\n", cal_mean(p, n));
	
	make_double(p, n);

	print_array(p, n);

	return 0;
}

void print_array(int lst[], int num)
{
	printf("List of array elements:\n");
	for(int i=0; i<num; i++)
	{
		printf("%d", lst[i]);
		if(i<num-1)
		{
			printf(", ");
		}
	}
	printf("\n");
}

float cal_mean(int lst[], int num)
{
	int sum = 0;
	
	for(int i=0; i<num; i++)
	{
		sum += lst[i];
	}

	return (float)sum / num;
}

void make_double(int lst[], int num) // call by reference
{
	for(int i=0; i<num; i++)
	{
		lst[i] *= 2;
	}
}
