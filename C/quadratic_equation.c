#include <stdio.h>
#include <math.h>
#define TRUE 1
#define FALSE 0

float CalculateDelta(float a, float b, float c);

int main()
{
	float a, b, c;
	printf("Quadratic Equation is like: ");
	printf("ax%c + bx + c = 0\n", 0xFD);
	printf("Please enter the coefficient a: ");
	scanf("%f", &a);

	printf("Please enter the coefficient b: ");
	scanf("%f", &b);

	printf("Please enter the coefficient c: ");
	scanf("%f", &c);

	float x1, x2;
	if(a == 0)
	{
		if(b==0)
		{
			printf("Error: the equation cannot be solved.\n");
		}
		else
		{
			x1 = -c/b;
			printf("This is a first-order equation!\n");
			printf("x = %.3f\n", x1);
		}
	}
	else
	{
		float delta = CalculateDelta(a, b, c);
		if(delta < 0)
		{
			if(b==0)
			{
				printf("This equation has two complex roots.\n");
				printf("x1 =  %.3fi\n", sqrt(-delta)/(2*a));
				printf("x2 = -%.3fi\n", sqrt(-delta)/(2*a));
			}
			else
			{
				printf("This equation has two complex roots.\n");
				printf("x1 = %.3f + %.3fi\n", -b/(2*a), sqrt(-delta)/(2*a));
				printf("x2 = %.3f - %.3fi\n", -b/(2*a), sqrt(-delta)/(2*a));
			}
		}
		else if(delta == 0)
		{
			x1 = -b/(2*a);
			printf("This equation has a real \"double\" root.\n");
			printf("x1, x2 = %.3f\n", x1);
		}
		else
		{	
			x1 = (-b + sqrt(delta))/(2*a);
			x2 = (-b - sqrt(delta))/(2*a);
			printf("This equation has two real roots!\n");
			printf("x1 = %.3f\n", x1);
			printf("x2 = %.3f\n", x2);
		}
	}

	return 0;
}

float CalculateDelta(float a, float b, float c)
{
	float delta;
	delta = pow(b, 2) - 4*a*c;

	return delta;
}