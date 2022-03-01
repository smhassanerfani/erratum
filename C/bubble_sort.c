#include <stdio.h>

void print_array(int lst[], int num);
void bubble_sort(int lst[], int num);
void swap(int lst[], int i, int j);

int main()
{
	int p[] = {2, 5, 3, 6, 4, 4, 0};
	int n = sizeof(p) / sizeof(p[0]);

	print_array(p, n);

	bubble_sort(p, n);
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

void bubble_sort(int lst[], int num)
{
	for(int i=0; i<num; i++)
	{
		int swapped = 0;
		for(int j=0; j<num-1; j++)
		{
			if(lst[j] > lst[j+1])
			{
				swap(lst, j, j+1);
				swapped = 1;
			}
		}
		if(!swapped) break;
	}
}

void swap(int lst[], int i, int j)
{
	int temp = lst[j];
	lst[j] = lst[j+1];
	lst[j+1] = temp;
}