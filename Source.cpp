# Matrix-Multiplication-in-C++
#include <iostream>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <cmath>

using namespace std;

int leafsize;

void take_data(vector<vector<int>> &a, vector<vector<int>> &b, int r1, int c1, int r2, int c2);
void multiplication(vector<vector<int>> &a, vector<vector<int>> &b, vector<vector<int>> &mult, int r1, int c1, int r2, int c2);
void display(vector<vector<int>> &mult, int r1, int c2);

void strassen(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, unsigned int tam);
unsigned int nextPowerOfTwo(int n);
void strassenR(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C,
	int tam);
void sum(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, int tam);
void subtract(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, int tam);

void printMatrix(vector< vector<int> > matrix, int n);

void ikjalgorithm(vector< vector<int> > A,
	vector< vector<int> > B,
	vector< vector<int> > &C, int n) {
	for (int i = 0; i < n; i++) {
		for (int k = 0; k < n; k++) {
			for (int j = 0; j < n; j++) {
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}
}

void strassenR(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, int tam) {
	if (tam <= leafsize) {
		ikjalgorithm(A, B, C, tam);
		return;
	}

	// other cases are treated here:
	else {
		int newTam = tam / 2;
		vector<int> inner(newTam);
		vector< vector<int> >
			a11(newTam, inner), a12(newTam, inner), a21(newTam, inner), a22(newTam, inner),
			b11(newTam, inner), b12(newTam, inner), b21(newTam, inner), b22(newTam, inner),
			c11(newTam, inner), c12(newTam, inner), c21(newTam, inner), c22(newTam, inner),
			p1(newTam, inner), p2(newTam, inner), p3(newTam, inner), p4(newTam, inner),
			p5(newTam, inner), p6(newTam, inner), p7(newTam, inner),
			aResult(newTam, inner), bResult(newTam, inner);

		int i, j;

		//dividing the matrices in 4 sub-matrices:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				a11[i][j] = A[i][j];
				a12[i][j] = A[i][j + newTam];
				a21[i][j] = A[i + newTam][j];
				a22[i][j] = A[i + newTam][j + newTam];

				b11[i][j] = B[i][j];
				b12[i][j] = B[i][j + newTam];
				b21[i][j] = B[i + newTam][j];
				b22[i][j] = B[i + newTam][j + newTam];
			}
		}

		// Calculating p1 to p7:

		sum(a11, a22, aResult, newTam); // a11 + a22
		sum(b11, b22, bResult, newTam); // b11 + b22
		strassenR(aResult, bResult, p1, newTam); // p1 = (a11+a22) * (b11+b22)

		sum(a21, a22, aResult, newTam); // a21 + a22
		strassenR(aResult, b11, p2, newTam); // p2 = (a21+a22) * (b11)

		subtract(b12, b22, bResult, newTam); // b12 - b22
		strassenR(a11, bResult, p3, newTam); // p3 = (a11) * (b12 - b22)

		subtract(b21, b11, bResult, newTam); // b21 - b11
		strassenR(a22, bResult, p4, newTam); // p4 = (a22) * (b21 - b11)

		sum(a11, a12, aResult, newTam); // a11 + a12
		strassenR(aResult, b22, p5, newTam); // p5 = (a11+a12) * (b22)   

		subtract(a21, a11, aResult, newTam); // a21 - a11
		sum(b11, b12, bResult, newTam); // b11 + b12
		strassenR(aResult, bResult, p6, newTam); // p6 = (a21-a11) * (b11+b12)

		subtract(a12, a22, aResult, newTam); // a12 - a22
		sum(b21, b22, bResult, newTam); // b21 + b22
		strassenR(aResult, bResult, p7, newTam); // p7 = (a12-a22) * (b21+b22)

		// calculating c21, c21, c11 e c22:

		sum(p3, p5, c12, newTam); // c12 = p3 + p5
		sum(p2, p4, c21, newTam); // c21 = p2 + p4

		sum(p1, p4, aResult, newTam); // p1 + p4
		sum(aResult, p7, bResult, newTam); // p1 + p4 + p7
		subtract(bResult, p5, c11, newTam); // c11 = p1 + p4 - p5 + p7

		sum(p1, p3, aResult, newTam); // p1 + p3
		sum(aResult, p6, bResult, newTam); // p1 + p3 + p6
		subtract(bResult, p2, c22, newTam); // c22 = p1 + p3 - p2 + p6

		// Grouping the results obtained in a single matrix:
		for (i = 0; i < newTam; i++) {
			for (j = 0; j < newTam; j++) {
				C[i][j] = c11[i][j];
				C[i][j + newTam] = c12[i][j];
				C[i + newTam][j] = c21[i][j];
				C[i + newTam][j + newTam] = c22[i][j];
			}
		}
	}
}

unsigned int nextPowerOfTwo(int n) {
	return pow(2, int(ceil(log2(n))));
}

void strassen(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, unsigned int n) {
	//unsigned int n = tam;
	unsigned int m = nextPowerOfTwo(n);
	vector<int> inner(m);
	vector< vector<int> > APrep(m, inner), BPrep(m, inner), CPrep(m, inner);

	for (unsigned int i = 0; i<n; i++) {
		for (unsigned int j = 0; j<n; j++) {
			APrep[i][j] = A[i][j];
			BPrep[i][j] = B[i][j];
		}
	}

	strassenR(APrep, BPrep, CPrep, m);
	for (unsigned int i = 0; i<n; i++) {
		for (unsigned int j = 0; j<n; j++) {
			C[i][j] = CPrep[i][j];
		}
	}
}

void sum(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, int tam) {
	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			C[i][j] = A[i][j] + B[i][j];
		}
	}
}

void subtract(vector< vector<int> > &A,
	vector< vector<int> > &B,
	vector< vector<int> > &C, int tam) {
	int i, j;

	for (i = 0; i < tam; i++) {
		for (j = 0; j < tam; j++) {
			C[i][j] = A[i][j] - B[i][j];
		}
	}
}




void printMatrix(vector< vector<int> > matrix, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			if (j != 0) {
				cout << "\t";
			}
			cout << matrix[i][j];
		}
		cout << endl;
	}
}


vector<vector<int>> Multiply(vector<vector<int>> matrixFirst, vector<vector<int>> matrixSecond, int dimensionsRow, int dimensionsColumn) {



	int rowFirst = dimensionsRow;
	int columnFirst = dimensionsColumn;

	int rowSecond = dimensionsRow;
	int columnSecond = dimensionsColumn;
	//Initialize resultation matrix

	vector<vector<int>> resultMatrix;
	vector<int> tempSecond;
	for (int i = 0; i<rowFirst; i++) {
		for (int j = 0; j<columnFirst; j++) {
			tempSecond.push_back(0);

		}
		resultMatrix.push_back(tempSecond);
		tempSecond.erase(tempSecond.begin(), tempSecond.end());
	}


	//Multiplying code
	for (int i = 0; i<rowFirst; i++) {
		for (int j = 0; j<columnSecond; j++) {
			for (int k = 0; k<rowFirst; k++) {
				resultMatrix[i][j] += matrixFirst[i][k] * matrixSecond[k][j];
			}
		}
	}
	return resultMatrix;
}

int main(int argc, char* argv[])
{
	int option = -1;

	cout << "Select Option:\n1: For Normal\n2: For Strassen: ";
	cin >> option;

	if (option == 1) {

		int r1, c1, r2, c2, i, j, k;
		//int a[10][10], b[10][10], mult[10][10],

		cout << "Enter rows and columns for first matrix: ";
		cin >> r1 >> c1;
		cout << "Enter rows and columns for second matrix: ";
		cin >> r2 >> c2;



		/* If column of first matrix is not equal to row of second matrix, asking user to enter the size of matrix again. */

		while (c1 != r2)
		{
			cout << "Error! column of first matrix not equal to row of second." << endl;
			cout << "Enter rows and columns for first matrix: ";
			cin >> r1 >> c1;
			cout << "Enter rows and columns for second matrix: ";
			cin >> r2 >> c2;
		}

		vector<vector<int>> a;
		vector<vector<int>> b;
		vector<vector<int>> mult;

		a.resize(r1);
		for (int i = 0; i < r1; i++) {
			a[i].resize(c1);
		}


		b.resize(r2);
		for (int i = 0; i < r2; i++) {
			b[i].resize(c2);
		}


		mult.resize(r1);
		for (int i = 0; i < r1; i++) {
			mult[i].resize(c2);
		}

		cout << "A row: " << a.size() << " A col: " << a[0].size() << endl;
		cout << "B row: " << b.size() << " B col: " << b[0].size() << endl;



		take_data(a, b, r1, c1, r2, c2);  /* Function to take matrices data */
		cout << "A: ";
		display(a, r1, c1);
		cout << "B: ";
		display(b, r2, c2);
		clock_t tStart = clock();
		multiplication(a, b, mult, r1, c1, r2, c2); /* Function to multiply two matrices. */
		double duration = (clock() - tStart) / (double)CLOCKS_PER_SEC;
		display(mult, r1, c2); /* Function to display resultant matrix after multiplication. */
		cout << "The time take by Normal Multiplication is: "<<duration;
		return 0;
	}

	if (option == 2) { // Strassen

		string filename;
		if (argc < 3) {
			filename = "2000.in";
		}
		else {
			filename = argv[2];
		}

		if (argc < 5) {
			leafsize = 16;
		}
		else {
			leafsize = atoi(argv[4]);
		}

		srand(time(NULL));
		int rows = 128;
		int column = rows;

		vector<vector<int>> A;
		vector<int> tempSecond;
		for (int i = 0; i<rows; i++) {
			for (int j = 0; j<column; j++) {
				tempSecond.push_back(rand() % 10);

			}
			A.push_back(tempSecond);
			tempSecond.erase(tempSecond.begin(), tempSecond.end());
		}

		vector<vector<int>> B;
		vector<int> tempFirst;
		for (int i = 0; i<rows; i++) {
			for (int j = 0; j<column; j++) {
				tempFirst.push_back(rand() % 10);

			}

			B.push_back(tempFirst);
			tempFirst.erase(tempFirst.begin(), tempFirst.end());

		}


		vector<vector<int>> C;
		vector<int> tempThird;
		for (int i = 0; i<rows; i++) {
			for (int j = 0; j<column; j++) {
				tempThird.push_back(rand() % 10);

			}

			C.push_back(tempThird);
			tempThird.erase(tempThird.begin(), tempThird.end());

		}



		int n = rows;
		clock_t tStart2 = clock();
		strassen(A, B, C, n);
		double duration2 = (clock() - tStart2) / (double)CLOCKS_PER_SEC;
		printMatrix(C, n);
		cout << "The time take by Strassen Multiplication is: " << duration2;



		vector<vector<int>> R = Multiply(A, B, rows, column);
		cout << endl;
		cout << endl;
		cout << endl;
		cout << endl;
		cout << endl;




		for (int i = 0; i<rows; i++) {
			for (int j = 0; j<column; j++) {
				cout << R[i][j] << "  ";

			}

			cout << endl;
		}



	}
}


void take_data(vector<vector<int>> &a, vector<vector<int>> &b, int r1, int c1, int r2, int c2)
{

	srand(time(NULL));
	int random = -1;

	int i, j;

	for (i = 0; i < a.size(); i++)
		for (j = 0; j < a[0].size(); j++)
		{
		
		random = rand() % 10 + 1;
		a[i][j] = random;


		}

	for (i = 0; i < b.size(); i++)
		for (j = 0; j < b[0].size(); j++)
		{
		random = rand() % 10 + 1;
		b[i][j] = random;

		}




}

void multiplication(vector<vector<int>> &a, vector<vector<int>> &b, vector<vector<int>> &mult, int r1, int c1, int r2, int c2)
{
	int i, j, k;
	/* Initializing elements of matrix mult to 0.*/
	for (i = 0; i<r1; ++i)
		for (j = 0; j<c2; ++j)
		{
		mult[i][j] = 0;
		}
	/* Multiplying matrix a and b and storing in array mult. */
	for (i = 0; i<r1; ++i)
		for (j = 0; j<c2; ++j)
			for (k = 0; k<c1; ++k)
			{
		mult[i][j] += a[i][k] * b[k][j];
			}
}

void display(vector<vector<int>> &mult, int r1, int c2)
{
	int i, j;
	cout << endl << "Output Matrix: " << endl;
	for (i = 0; i<r1; ++i)
		for (j = 0; j<c2; ++j)
		{
		cout << " " << mult[i][j];
		if (j == c2 - 1)
			cout << endl;
		}
}


