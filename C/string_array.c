#include <stdio.h>

int main()
{	
	printf("Please enter your name: \n");
	char name[10];
	scanf("%s", name);

	printf("your name is %s\n", name);

	int N = sizeof(name) / sizeof(name[0]);
	printf("Length of array: %d\n", N);
	
	int M = 0;
	for(int i=0; i<N; i++)
	{
		if(name[i] == '\0') break;
		M++;
		printf("%c\n", name[i]);
	}

	printf("Length of name: %d\n", M);
	return 0;
}