#include <stdio.h>

/*
x++ increments the value of variable x after processing the current statement.
++x increments the value of variable x before processing the current statement.
x += ++i will increment i and add i+1 to x. x += i++ will add i to x, then increment i.
*/

int main(){
	int a1, a2;
	int b1, b2;

	a1 = 10;
	a2 = 10;

	b1 = a1++;
	b2 = ++a2;

	printf("a1 = %d\n", a1);
	printf("a2 = %d\n", a2);

	printf("b1 = %d\n", b1);
	printf("b2 = %d\n", b2);

	return 0;
}
