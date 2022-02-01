#include <stdio.h>
#define TRUE -1

int main()
{
	char g;
	int nA = 0;
	int nB = 0;
	int nC = 0;
	int nD = 0;
	int nF = 0;

	// while(TRUE)
	// {
	// 	g = getchar();
	// 	if(g == EOF) break;
	// }

	while((g = getchar()) != EOF) //EOF (End Of File): ctrl+d
	{
		switch(g)
		{
			case 'A':
			case 'a':
				nA++;
				break;

			case 'B':
			case 'b':
				nB++;
				break;

			case 'C':
			case 'c':
				nC++;
				break;

			case 'D':
			case 'd':
				nD++;
				break;

			case 'F':
			case 'f':
				nF++;
				break;

			case ' ':
			case '\n':
			case '\t':
				break;
			
			default:
				printf("Incorrect input. Please enter from the letters set {A, B, C, D, F}.\n");
		} // end of switch(g)
	} // end of while

	int n = nA + nB + nC + nD + nF;
	int cA = nA;
	int cB = cA + nB;
	int cC = cB + nC;
	int cD = cC + nD;
	int cF = cD + nF;
	printf("Grade\t Count\t\t Percent\t Cum. Percent\n");
	printf("A\t %d of %d\t %.2f\t\t %.2f\n", nA, n, (float)nA*100/n, (float)cA*100/n);
	printf("B\t %d of %d\t %.2f\t\t %.2f\n", nB, n, (float)nB*100/n, (float)cB*100/n);
	printf("C\t %d of %d\t %.2f\t\t %.2f\n", nC, n, (float)nC*100/n, (float)cC*100/n);
	printf("D\t %d of %d\t %.2f\t\t %.2f\n", nD, n, (float)nD*100/n, (float)cD*100/n);
	printf("F\t %d of %d\t %.2f\t\t %.2f\n", nF, n, (float)nF*100/n, (float)cF*100/n);

	if((float)cC/n > 0.6)
	{
		printf("congratulations to Teacher!\n");
	}

	return 0;	
}