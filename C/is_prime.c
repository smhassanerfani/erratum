#include <stdio.h>
#include <math.h>
#define TRUE 1
#define FALSE 0

int IsPrime(int a);

int main()
{
	int a;
	

	printf("Enter your number: ");
	scanf("%d", &a);
	printf("\n");

	if(IsPrime(a))
	{
		printf("%d is a prime number.\n", a);
	}
	else
	{
		printf("%d is NOT a prime number.\n", a);	
	}

	return 0;
}

int IsPrime(int a)
{
	int prime = TRUE;
	for(int i = 2; i <= sqrt(a); i++)
	{
		if(a % i == 0)
		{
			prime = FALSE;
			break;
		}
	}
	return prime; 	
}