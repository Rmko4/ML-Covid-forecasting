#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STRLEN 30
#define DATELEN 9

typedef struct matrixstruct {
  int rows, columns;
  double **matrix;
} * Matrix;

void *safeMalloc(int n) {
  void *ptr = malloc(n);
  if (ptr == NULL) {
    perror("Allocation failed.\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

Matrix makeMatrix(int r, int c) {
  Matrix A = safeMalloc(sizeof(struct matrixstruct));
  A->rows = r;
  A->columns = c;
  A->matrix = safeMalloc(r * sizeof(double *));
  for (int row = 0; row < r; row++) {
    A->matrix[row] = safeMalloc(c * sizeof(double));
  }
  return A;
}

void freeMatrix(Matrix A) {
  for (int row = 0; row < A->rows; row++) {
    free(A->matrix[row]);
  }
  free(A->matrix);
  free(A);
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

Matrix readDataFile(char *fileName) {
  int nSeries, nObsv, nTimeSteps, col;
  char *p, **names, **dates;
  Matrix A;
  FILE *fp;

  fp = fopen(fileName, "r");

  if (fp == NULL) {
    perror("Error while opening the file.\n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", &nSeries);
  fscanf(fp, "%d", &nObsv);
  fscanf(fp, "%d", &nTimeSteps);

  names = safeMalloc(nSeries * sizeof(char *));

  for (int i = 0; i < nSeries; i++) {
    names[i] = safeMalloc(STRLEN * sizeof(char));
    fscanf(fp, "%s", names[i]);
  }

  dates = safeMalloc(nSeries * sizeof(char *));

  for (int i = 0; i < nSeries; i++) {
    dates[i] = safeMalloc(DATELEN * sizeof(char));
    fscanf(fp, "%s", dates[i]);
  }

  col = nSeries * nObsv;
  A = makeMatrix(nTimeSteps, col);

  for (int r = 0; r < nTimeSteps; r++) {
    for (int c = 0; c < col; c++) {
      fscanf(fp, "%lf", &A->matrix[r][c]);
    }
  }

  for (int r = 0; r < nTimeSteps; r++) {
    for (int c = 0; c < col; c++) {
      printf("%lf ", A->matrix[r][c]);
    }
    printf("\n");
  }

  fclose(fp);

  return A;
}

int main(int argc, char *argv[]) {
  int windowSize;
  long strParse;
  char *end, *fileName;
  Matrix A;

  if (argc < 2) {
    printf("Provide args: <filename> <window length>");
    exit(EXIT_FAILURE);
  }

  errno = 0;
  fileName = argv[1];
  strParse = strtol(argv[2], &end, 10);

  if (errno != 0 || *end != '\0' || strParse > INT_MAX) {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }

  windowSize = strParse;

  A = readDataFile(fileName);
  freeMatrix(A);

  if (windowSize > A->rows - 1) {
    printf("Length of the window is too large.");
    exit(EXIT_FAILURE);
  }

  return 0;
}
