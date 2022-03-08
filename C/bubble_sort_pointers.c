#include <stdio.h>

void print_array(const int *const lst, int num);
void bubble_sort(int *const lst, int num, int (*compare)(int a, int b));
void swap(int *aPtr, int *bPtr);
int ascending(int a, int b);
int descending(int a, int b);
int ascending_rem3(int a, int b);

int main()
{
	int p[] = {2, 5, 3, 6, 4, 4, 0};
	int n = sizeof(p) / sizeof(p[0]);

	print_array(p, n);

	int (*f[3])(int, int) = {ascending, descending, ascending_rem3};
	bubble_sort(p, n, f[0]);
	print_array(p, n);

	return 0;
}

void print_array(const int *const lst, int num)
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

void bubble_sort(int *const lst, int num, int (*compare)(int a, int b))
{
	for(int i=0; i<num; i++)
	{
		int swapped = 0;
		for(int j=0; j<num-1; j++)
		{
			if(!(*compare)(lst[j], lst[j+1]))
			{
				swap(&lst[j], &lst[j+1]);
				swapped = 1;
			}
		}
		if(!swapped) break;
	}
}

void swap(int *aPtr, int *bPtr)
{
	int temp = *bPtr;
	*bPtr = *aPtr;
	*aPtr = temp;
}

int ascending(int a, int b)
{
	return (a <= b);
}

int descending(int a, int b)
{
	return (a >= b);
}

int ascending_rem3(int a, int b)
{
	return ((a%3) <= (b%3));
}