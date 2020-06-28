#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define STRLEN 30
#define DATELEN 9
#define LAYERS 3

typedef struct matrixstruct {
  int rows, columns;
  double **matrix;
} * Matrix;

typedef struct weightstruct {
  int layers; // Amount of layers in mlp
  int *size;  // size per layer
  Matrix *W;  // Weights matrices
  double **b; // Bias vectors
} Weight;

typedef struct mlpstruct {
  double **x; // activation
  Weight weight;
} MLP;

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

MLP makeMLP(int layers, int *size) {
  MLP model;
  Weight weight;

  weight = model.weight;
  weight.layers = layers;
  weight.size = size;

  weight.W = safeMalloc((layers - 1) * sizeof(Matrix));
  weight.b = safeMalloc((layers - 1) * sizeof(double));
  model.x = safeMalloc(layers * sizeof(double));

  // Weight layer k is at k - 1
  for (int k = 0; k < layers - 1; k++) {
    weight.W[k] = makeMatrix(size[k + 1], size[k]);
    weight.b[k] = safeMalloc(size[k + 1]);
    model.x = safeMalloc(size[k] * sizeof(double));
  }
  // Output layer activation units.
  model.x = safeMalloc(size[layers - 1] * sizeof(double));
}

void freeMatrix(Matrix A) {
  free(A->matrix[0]);
  free(A->matrix);
  free(A);
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

Matrix *readDataFile(char *fileName, int *nSeries, int *nVariate,
                     char ***names) {
  int nTimeSteps;
  Matrix *Sample;
  FILE *fp;

  fp = fopen(fileName, "r");

  if (fp == NULL) {
    perror("Error while opening the file.\n");
    exit(EXIT_FAILURE);
  }

  fscanf(fp, "%d", nSeries);
  fscanf(fp, "%d", nVariate);

  *names = safeMalloc(*nSeries * sizeof(char *));
  Sample = safeMalloc(*nSeries * sizeof(Matrix));

  for (int i = 0; i < *nSeries; i++) {
    (*names)[i] = safeMalloc(STRLEN * sizeof(char));
    fscanf(fp, "%s", (*names)[i]);
    fscanf(fp, "%d", &nTimeSteps);

    Sample[i] = makeMatrix(nTimeSteps, *nVariate);

    for (int r = 0; r < nTimeSteps; r++) {
      for (int c = 0; c < *nVariate; c++) {
        fscanf(fp, "%lf", &(Sample[i]->matrix[r][c]));
      }
    }
  }

  fclose(fp);

  return Sample;
}

int intParse(char *arg) {
  char *end;
  long strParse;

  strParse = strtol(arg, &end, 10);
  errno = 0;

  if (errno != 0 || *end != '\0' || strParse > INT_MAX) {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }
  return (int)strParse;
}

int main(int argc, char *argv[]) {
  int windowSize, nSeries, nVariate;
  int size[LAYERS], nHiddenUnits;
  char *fileName, **names;

  Matrix *sample;
  MLP model;

  if (argc < 3) {
    printf("Provide args: <filename> <window length> <number hidden units>");
    exit(EXIT_FAILURE);
  }

  windowSize = intParse(argv[2]);
  nHiddenUnits = intParse(argv[3]);

  sample = readDataFile(fileName, &nSeries, &nVariate, &names);

  size[0] = nVariate * windowSize;
  size[1] = nHiddenUnits;
  size[2] = nVariate;

  model = makeMLP(LAYERS, size);

  for (int i = 0; i < nSeries; i++) {
    freeMatrix(sample[i]);
  }
  free(sample);
  // if (windowSize > A->rows - 1) {
  //  printf("Length of the window is too large.");
  //  exit(EXIT_FAILURE);
  //}

  /*for (int i = 0; i < nSeries; i++) {
    printf("%s\n", names[i]);
    for (int r = 0; r < sample[i]->rows; r++) {
      for (int c = 0; c < sample[i]->columns; c++) {
        printf("%lf ", sample[i]->matrix[r][c]);
      }
      printf("\n");
    }
  }*/
  return 0;
}
