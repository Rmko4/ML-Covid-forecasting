#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STRLEN 30
#define LAYERS 3
#define MAXA 1E3
#define ITER 1000

typedef struct matrixstruct {
  int rows, columns;
  double **matrix;
} * Matrix;

typedef struct weightstruct {
  int layers; // Amount of layers in mlp
  int *size;
  Matrix *W;  // Weights matrices
  double **b; // Bias vectors
} * Weight;

typedef struct mlpstruct {
  double **x; // activation
  Weight weight;
} * MLP;

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

double randRange(double min, double max) {
  double div = RAND_MAX / (max - min);
  return min + rand() / div;
}

Weight makeWeight(int layers, int *size) {
  Weight weight = safeMalloc(sizeof(struct weightstruct));

  weight->layers = layers;
  weight->size = size;
  weight->W = safeMalloc((layers - 1) * sizeof(Matrix));
  weight->b = safeMalloc((layers - 1) * sizeof(double *));

  // Weight layer k is at k - 1
  for (int k = 0; k < layers - 1; k++) {
    weight->W[k] = makeMatrix(size[k + 1], size[k]);
    weight->b[k] = safeMalloc(size[k + 1] * sizeof(double));
  }
  return weight;
}

void freeWeight(Weight weight) {
  for (int k = 0; k < weight->layers; k++) {
    freeMatrix(weight->W[k]);
    free(weight->b[k]);
  }
  free(weight->W);
  free(weight->b);
  free(weight);
}

// Requires seeded rand.
void initWeight(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  double *b;

  layers = weight->layers;

  for (int k = 0; k < layers - 1; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        W->matrix[i][j] = randRange(-0.1, 0.1); // Change range
      }
      b[i] = 0;
    }
  }
}

// Expects weights of equal size configuration.
void addScaledWeight(Weight weightTo, Weight weightFrom, int a, int b) {
  int layers, rows, columns;
  Matrix WTo, WFrom;
  double *bTo, *bFrom;

  layers = weightTo->layers;

  for (int k = 0; k < layers - 1; k++) {
    WTo = weightTo->W[k];
    bTo = weightTo->b[k];
    WFrom = weightFrom->W[k];
    bFrom = weightFrom->b[k];
    rows = WTo->rows;
    columns = WTo->columns;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        WTo->matrix[i][j] +=
            a * WFrom->matrix[i][j] - b * WTo->matrix[i][j]; // Change range
      }
      bTo[i] += a * bFrom[i] - a * bTo[i];
    }
  }
}

void setZeroWeight(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  double *b;

  layers = weight->layers;

  for (int k = 0; k < layers - 1; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        W->matrix[i][j] = 0;
      }
      b[i] = 0;
    }
  }
}

double l2Norm(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  double *b, l2n;

  layers = weight->layers;
  l2n = 0;

  for (int k = 0; k < layers - 1; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        l2n += W->matrix[i][j] * W->matrix[i][j];
      }
      l2n += b[i] * b[i];
    }
  }

  return l2n;
}

MLP makeMLP(int layers, int *size) {
  MLP model = safeMalloc(sizeof(struct mlpstruct));
  model->weight = makeWeight(layers, size);
  model->x = safeMalloc(layers * sizeof(double *));

  for (int k = 0; k < layers; k++) {
    model->x[k] = safeMalloc(size[k] * sizeof(double));
  }
  return model;
}

void freeMLP(MLP model) {
  int layers = model->weight->layers;
  for (int k = 0; k < layers; k++) {
    free(model->x[k]);
  }
  free(model->x);
  freeWeight(model->weight);
  free(model);
}

double **flattenSample(Matrix *sample, int nSeries, int *len, int window,
                       int outIdx) {
  int rows, n, t, i;
  double **flatSample;

  i = 0;
  *len = -(window + 1) * (nSeries - 1) - sample[outIdx]->rows;

  for (n = 0; n < nSeries; n++) {
    *len += sample[n]->rows - (window + 1);
  }

  flatSample = safeMalloc(*len * sizeof(double *));

  for (n = 0; n < nSeries; n++) {
    rows = sample[n]->rows - (window + 1);
    for (t = 0; t < rows; t++) {
      flatSample[i] = &sample[n]->matrix[t][0];
      i++;
    }
  }

  return flatSample;
}

// Expects len > 1
void shuffleSample(double **a, int len) {
  int i, j;
  double *t;
  for (i = 0; i < len - 1; i++) {
    j = i + rand() / (RAND_MAX / (len - i) + 1);
    t = a[j];
    a[j] = a[i];
    a[i] = t;
  }
}

double sigmoid(double x) { return 1 / (1 + exp(-x)); }

// Partial derivative of MSE
double dMSE(double yhat, double y) { return 2 * (yhat - y); }

// L2 Loss
double MSE(double *yhat, double *y, int len) {
  int i;
  double mse, err;
  mse = 0;
  for (i = 0; i < len; i++) {
    err = yhat[i] - y[i];
    mse += err * err;
  }
  return mse / len;
}

double mean(double *a, int len) {
  int m, i;
  m = 0;
  for (i = 0; i < len; i++) {
    m += a[i];
  }
  return m /= len;
}

int argmin(double *a, int len) {
  int i, imin, min;
  imin = 0;
  min = DBL_MAX;
  for (i = 0; i < len; i++) {
    imin = a[i] < min ? i : imin;
  }
  return imin;
}

double *forwardMLP(MLP model, double *u) {
  int layers, rows, columns, i, j, k;
  double *b, **x;
  Matrix W;
  Weight weight;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;

  // Set input layer with no function applied.
  columns = weight->W[0]->columns;
  for (i = 0; i < columns; i++) {
    x[0][i] = u[i];
  }

  // For every hidden layer (with sigmoid).
  for (k = 0; k < layers - 2; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    for (i = 0; i < rows; i++) {
      x[k + 1][i] = b[i];
      for (j = 0; j < columns; j++) {
        x[k + 1][i] += W->matrix[i][j] * x[k][j];
      }
      x[k + 1][i] = sigmoid(x[k + 1][i]);
    }
  }

  // Output layer (no function)
  k = layers - 2;
  W = weight->W[k];
  b = weight->b[k];
  rows = W->rows;
  columns = W->columns;

  for (i = 0; i < rows; i++) {
    x[k + 1][i] = b[i];
    for (j = 0; j < columns; j++) {
      x[k + 1][i] += W->matrix[i][j] * x[k][j];
    }
  }

  return x[layers - 1];
}

Weight backPropMLP(MLP model, double *y, Weight gradient) {
  int layers, rows, columns, i, j, k;
  double *b, **x, **delta;
  Matrix W;
  Weight weight;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;

  delta = safeMalloc(layers * sizeof(double *));

  k = layers - 2;
  W = weight->W[k];
  rows = W->rows;
  delta[k] = safeMalloc(rows * sizeof(double));
  for (i = 0; i < rows; i++) {
    delta[k][i] = dMSE(x[k + 1][i], y[i]);
  }

  for (k = layers - 2; k > 0; k--) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    delta[k - 1] = safeMalloc(columns * sizeof(double));

    for (i = 0; i < columns; i++) {
      delta[k - 1][i] = 0;
      for (j = 0; j < rows; j++) {
        delta[k - 1][i] += delta[k][j] * W->matrix[j][i];
      }
      delta[k - 1][i] *= x[k][i] * (1 - x[k][i]);
    }
  }

  // Calculating final gradient
  for (k = 0; k < layers - 1; k++) {
    W = gradient->W[k];
    b = gradient->b[k];
    rows = W->rows;
    columns = W->columns;
    for (i = 0; i < rows; i++) {
      for (j = 0; j < columns; j++) {
        W->matrix[i][j] += delta[k][j] * x[k][i];
      }
      b[i] += delta[k][i];
    }
  }

  for (k = 0; k < layers - 1; k++) {
    free(delta[k]);
  }
  free(delta);
  return gradient;
}

double meanRisk(MLP model, Matrix val, int window, int out) {
  int i, len;
  double *yhat, risk;

  // Matrix valJ = sample[j];
  len = val->rows - window;
  risk = 0;
  // Calculate validation risk on j.
  for (i = 0; i < len; i++) {
    yhat = forwardMLP(model, val->matrix[i]);
    risk += MSE(yhat, val->matrix[i + window], out);
  }
  risk /= len;
}

void trainMLP(MLP model, Matrix *sample, int nSeries, int window, int maxIter,
              double mu) {
  double upA, maxA, a;
  double *b, **x, **delta, **S;
  double *riskR, *riskJ;
  int layers, rows, columns;
  int scaleGrad, scaleReg, sampleSize;
  int in, out, r, i, j, n;
  Matrix W;
  Weight weight, gradient;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;
  in = weight->size[0];
  out = weight->size[layers - 1];

  gradient = makeWeight(layers, weight->size);
  riskR = safeMalloc(maxIter * sizeof(double)); // MaxIter is now the same.
  riskJ = safeMalloc(maxIter * sizeof(double));

  maxA = log(MAXA);
  upA = maxA / maxIter;
  a = 0;

  // Iterating over degrees of flexibility r
  // r represents linear increases log alpha
  for (r = 0; r < maxIter; r++) {
    printf("\n############################\n");
    printf("Regularization - ALPHA: = %.3e\n", exp(a));
    printf("############################\n");
    // Every iteration leave out one complete series.
    for (j = 0; j < nSeries; j++) {
      printf("## Fold - K: %d\n", j);
      S = flattenSample(sample, nSeries, &sampleSize, window, j);
      scaleGrad = -mu / sampleSize;
      scaleReg = -2 * (exp(a) * exp(a)) / sampleSize;
      // Gradient descent iterations
      initWeight(model->weight);
      for (n = 0; n < ITER; n++) {
        printf("Epoch: %d\n", n);
        setZeroWeight(gradient);
        shuffleSample(S, sampleSize);
        for (i = 0; i < sampleSize; i++) {
          forwardMLP(model, S[i]);
          backPropMLP(model, S[i + in], gradient);
        }
        addScaledWeight(weight, gradient, scaleGrad, scaleReg);
      }

      riskJ[j] = meanRisk(model, sample[j], window, out);
      free(S);
    }

    // Calculate average risk for regularization r
    riskR[r] = mean(riskJ, nSeries);

    printf(" - Average risk: %.5lf - \n", riskR[r]);
    a += upA; // Increase the log alpha
  }

  r = argmin(riskR, maxIter); // regularization with minimum average risk

  S = flattenSample(sample, nSeries, &sampleSize, window, -1);
  scaleGrad = -mu / sampleSize;
  scaleReg = -2 * (exp(a) * exp(a)) / sampleSize;
  // Gradient descent iterations
  initWeight(model->weight);
  for (n = 0; n < ITER; n++) {
    setZeroWeight(gradient);
    shuffleSample(S, sampleSize);
    for (i = 0; i < sampleSize; i++) {
      forwardMLP(model, S[i]);
      backPropMLP(model, S[i + in], gradient);
    }
    addScaledWeight(weight, gradient, scaleGrad, scaleReg);
  }

  freeWeight(gradient);
  free(riskR);
  free(riskJ);
  free(S);
}

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

  fileName = argv[1];
  windowSize = intParse(argv[2]);
  nHiddenUnits = intParse(argv[3]);

  srand(time(NULL));

  sample = readDataFile(fileName, &nSeries, &nVariate, &names);

  size[0] = nVariate * windowSize;
  size[1] = nHiddenUnits;
  size[2] = nVariate;

  model = makeMLP(LAYERS, size);

  trainMLP(model, sample, nSeries, windowSize, ITER,
           0.005); // Implement variable learning rate

  double *yhat;
  yhat = forwardMLP(model, sample[0]->matrix[0]);
  for (int i = 0; i < nVariate; i++) {
    printf("yhat: %.5lf --- y: %.5lf\n", yhat[i],
           sample[0]->matrix[0 + windowSize][i]);
  }

  for (int i = 0; i < nSeries; i++) {
    freeMatrix(sample[i]);
  }
  free(sample);
  free(model);
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
