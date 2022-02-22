#include <stdio.h>

void print_array(int lst[], int num);
void get_divisors(int list[], int num);
int divisors_counter(int num);

int main()
{
	int num;
	printf("Please enter an integer number:");
	scanf("%d", &num);
	printf("\n");

	int num_of_divisors = divisors_counter(num);
	int list_of_divisors[num_of_divisors]; 
	
	printf("Number of divisors: %d\n", num_of_divisors);
	
	get_divisors(list_of_divisors, num);
	print_array(list_of_divisors, num_of_divisors);

	return 0;
}

void print_array(int lst[], int num)
{
	printf("List of divisor elements:\n");
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


int divisors_counter(int num)
{
	if(num==0) return 0;
	if(num<0) num = -num;

	int counter = 0;
	for(int i=1; i<=num; i++)
	{
		if(num % i == 0) counter++;
	}
	return counter;
}


void get_divisors(int list[], int num)
{
	int counter = 0;
	for(int i=1; i<=num; i++)
	{
		if(num % i == 0) 
		{
			list[counter] = i;
			counter++;
		}
	}
}
