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
  A->matrix[0] = safeMalloc(r * c * sizeof(double));
  for (int row = 1; row < r; row++) {
    A->matrix[row] = A->matrix[row - 1] + c;
  }
  return A;
}

void freeMatrix(Matrix A) {
  free(A->matrix[0]);
  free(A->matrix);
  free(A);
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

Matrix *readDataFile(char *fileName, int *nSeries, char ***names) {
  int nVariate, nTimeSteps;
  Matrix *Sample;
  FILE *fp;

  fp = fopen(fileName, "r");

  if (fp == NULL) {
    perror("Error while opening the file.\n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", nSeries);
  fscanf(fp, "%d", &nVariate);

  *names = safeMalloc(*nSeries * sizeof(char *));
  Sample = safeMalloc(*nSeries * sizeof(Matrix));

  for (int i = 0; i < *nSeries; i++) {
    (*names)[i] = safeMalloc(STRLEN * sizeof(char));
    fscanf(fp, "%s", (*names)[i]);
    fscanf(fp, "%d", &nTimeSteps);

    Sample[i] = makeMatrix(nTimeSteps, nVariate);

    for (int r = 0; r < nTimeSteps; r++) {
      for (int c = 0; c < nVariate; c++) {
        fscanf(fp, "%lf", &(Sample[i]->matrix[r][c]));
      }
    }
  }

  fclose(fp);

  return Sample;
}

int main(int argc, char *argv[]) {
  int windowSize, nUnits, nSeries;
  long strParse;
  char **names;
  char *end, *fileName;
  Matrix *sample;

  if (argc < 3) {
    printf("Provide args: <filename> <window length> <number hidden units>");
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
  strParse = strtol(argv[3], &end, 10);

  if (errno != 0 || *end != '\0' || strParse > INT_MAX || strParse < 1) {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }

  nUnits = strParse;

  sample = readDataFile(fileName, &nSeries, &names);

  for (int i = 0; i < nSeries; i++) {
    printf("%s\n", names[i]);
    for (int r = 0; r < sample[i]->rows; r++) {
      for (int c = 0; c < sample[i]->columns; c++) {
        printf("%lf ", sample[i]->matrix[r][c]);
      }
      printf("\n");
    }
  }

  for (int i = 0; i < nSeries; i++) {
    freeMatrix(sample[i]);
  }
  free(sample);

  //if (windowSize > A->rows - 1) {
  //  printf("Length of the window is too large.");
  //  exit(EXIT_FAILURE);
  //}

  return 0;
}
