#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define STRLEN 30
#define LAYERS 3
#define MAXA 5E-4
#define ITER 25000
#define REGITER 1

typedef struct matrixstruct {
  int rows, columns;
  float **matrix;
} * Matrix;

typedef struct weightstruct {
  int layers; // Amount of layers in mlp
  int *size;
  Matrix *W; // Weights matrices
  float **b; // Bias vectors
} * Weight;

typedef struct mlpstruct {
  float **x; // activation
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
  A->matrix = safeMalloc(r * sizeof(float *));
  A->matrix[0] = safeMalloc(r * c * sizeof(float));
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

float randRange(float min, float max) {
  float div = RAND_MAX / (max - min);
  return min + rand() / div;
}

Weight makeWeight(int layers, int *size) {
  Weight weight = safeMalloc(sizeof(struct weightstruct));

  weight->layers = layers;
  weight->size = size;
  weight->W = safeMalloc((layers - 1) * sizeof(Matrix));
  weight->b = safeMalloc((layers - 1) * sizeof(float *));

  // Weight layer k is at k - 1
  for (int k = 0; k < layers - 1; k++) {
    weight->W[k] = makeMatrix(size[k + 1], size[k]);
    weight->b[k] = safeMalloc(size[k + 1] * sizeof(float));
  }
  return weight;
}

void freeWeight(Weight weight) {
  for (int k = 0; k < weight->layers - 1; k++) {
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
  float *b, range;

  layers = weight->layers;

  for (int k = 0; k < layers - 1; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;
    range = sqrtf((float)6 / (rows + columns));
    if (k == 0) {
      range /= 5;
    }
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < columns; j++) {
        W->matrix[i][j] = randRange(-range, range); // Change range
      }
      b[i] = 0;
    }
  }
}

// Expects weights of equal size configuration.
void addScaledWeight(Weight weightTo, Weight weightFrom, float a, float b) {
  int layers, rows, columns;
  Matrix WTo, WFrom;
  float *bTo, *bFrom;

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
        WTo->matrix[i][j] += a * WFrom->matrix[i][j] - b * WTo->matrix[i][j];
      }
      bTo[i] += a * bFrom[i] - a * bTo[i];
    }
  }
}

void setZeroWeight(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  float *b;

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

void printWeight(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  float *b;

  layers = weight->layers;

  for (int k = 0; k < layers - 1; k++) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;
    printf("\n### Layer %d: ###\n", k);
    for (int j = 0; j < columns; j++) {
      printf("Unit %d: ", j);
      for (int i = 0; i < rows; i++) {
        printf("%.4f ", W->matrix[i][j]);
      }
      printf("\n");
    }
    printf("Bias: ");
    for (int i = 0; i < rows; i++) {
      printf("%.4f ", b[i]);
    }
    printf("\n");
  }
}

float l2Norm(Weight weight) {
  int layers;
  int rows, columns;
  Matrix W;
  float *b, l2n;

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
  model->x = safeMalloc(layers * sizeof(float *));

  for (int k = 0; k < layers; k++) {
    model->x[k] = safeMalloc(size[k] * sizeof(float));
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

float **flattenSample(Matrix *sample, int nSeries, int *len, int window,
                      int outIdx) {
  int rows, n, t, i;
  float **flatSample;

  i = 0;
  *len = -window * nSeries;
  if (outIdx != -1) {
    *len -= sample[outIdx]->rows - window;
  }

  for (n = 0; n < nSeries; n++) {
    *len += sample[n]->rows;
  }

  flatSample = safeMalloc(*len * sizeof(float *));

  for (n = 0; n < nSeries; n++) {
    if (n != outIdx) {
      rows = sample[n]->rows - (window);
      for (t = 0; t < rows; t++) {
        flatSample[i] = &sample[n]->matrix[t][0];
        i++;
      }
    }
  }

  return flatSample;
}

// Expects len > 1
void shuffleSample(float **a, int len) {
  int i, j;
  float *t;
  for (i = 0; i < len - 1; i++) {
    j = i + rand() / (RAND_MAX / (len - i) + 1);
    t = a[j];
    a[j] = a[i];
    a[i] = t;
  }
}

float sigmoid(float x) { return 1 / (1 + exp(-x)); }

// Partial derivative of MSE
float dMSE(float yhat, float y) { return 2 * (yhat - y); }

// L2 Loss
float MSE(float *yhat, float *y, int len) {
  int i;
  float mse, err;
  mse = 0;
  for (i = 0; i < len; i++) {
    err = yhat[i] - y[i];
    mse += err * err;
  }
  return mse / len;
}

float mean(float *a, int len) {
  int i;
  float m;
  m = 0;
  for (i = 0; i < len; i++) {
    m += a[i];
  }
  return m /= len;
}

int argmin(float *a, int len) {
  int i, imin;
  float min;
  imin = 0;
  min = FLT_MAX;
  for (i = 0; i < len; i++) {
    if (a[i] < min) {
      min = a[i];
      imin = i;
    }
  }
  return imin;
}

float *forwardMLP(MLP model, float *u) {
  int layers, rows, columns, i, j, k;
  float *b, **x;
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

Weight backPropMLP(MLP model, float *y, Weight gradient) {
  int layers, rows, columns, i, j, k;
  float *b, **x, **delta;
  Matrix W;
  Weight weight;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;

  delta = safeMalloc((layers - 1) * sizeof(float *));

  k = layers - 2;
  W = weight->W[k];
  rows = W->rows;
  delta[k] = safeMalloc(rows * sizeof(float));
  for (i = 0; i < rows; i++) {
    delta[k][i] = dMSE(x[k + 1][i], y[i]);
  }

  for (k = layers - 2; k > 0; k--) {
    W = weight->W[k];
    b = weight->b[k];
    rows = W->rows;
    columns = W->columns;

    delta[k - 1] = safeMalloc(columns * sizeof(float));

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
        W->matrix[i][j] += delta[k][i] * x[k][j];
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

float meanRisk(MLP model, Matrix val, int window, int nVariate) {
  int i, len;
  float *yhat, risk;

  // Matrix valJ = sample[j];
  len = val->rows - window;
  risk = 0;
  // Calculate validation risk on j.
  for (i = 0; i < len; i++) {
    yhat = forwardMLP(model, val->matrix[i]);
    risk += MSE(yhat, val->matrix[i + window], nVariate);
  }
  risk /= len;
}

void trainMLP(MLP model, Matrix *sample, int nSeries, int window, int maxIter,
              float mu) {
  float upA, maxA, a;
  float *b, **x, **delta, **S;
  float *riskR, *riskJ;
  float scaleGrad, scaleReg;
  int layers, rows, columns, sampleSize;
  int nVariate, r, i, j, n;
  Matrix W;
  Weight weight, gradient;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;
  nVariate = weight->size[layers - 1];

  gradient = makeWeight(layers, weight->size);
  riskR = safeMalloc(REGITER * sizeof(float)); // MaxIter is now the same.
  riskJ = safeMalloc(maxIter * sizeof(float));

  maxA = sqrt(MAXA);
  upA = maxA / REGITER;
  a = 0;

  // Iterating over degrees of flexibility r
  // r represents linear increases log alpha
  /*for (r = 0; r < REGITER; r++) {
    printf("\n###################################\n");
    printf("Regularization - ALPHA^2: = %.3e\n", a * a);
    printf("#####################################\n");
    // Every iteration leave out one complete series.
    for (j = 0; j < nSeries; j++) {
      printf("## Fold - K: %d\n", j);
      S = flattenSample(sample, nSeries, &sampleSize, window, j);
      scaleGrad = -mu / sampleSize;
      scaleReg = -2 * a * a;
      // Gradient descent iterations
      initWeight(model->weight);
      for (n = 0; n < ITER; n++) {
        // printf("Epoch: %d\n", n);
        setZeroWeight(gradient);
        shuffleSample(S, sampleSize);
        for (i = 0; i < sampleSize; i++) {
          forwardMLP(model, S[i]);
          backPropMLP(model, &S[i][nVariate * window], gradient);
        }
        addScaledWeight(weight, gradient, scaleGrad, scaleReg);
      }

      riskJ[j] = meanRisk(model, sample[j], window, nVariate);
      free(S);
    }

    // Calculate average risk for regularization r
    riskR[r] = mean(riskJ, nSeries);

    printf(" - Average risk: %f - \n", riskR[r]);
    a += upA; // Increase the log alpha
  }*/

  // r = argmin(riskR, REGITER); // regularization with minimum average risk
  // a = r * upA;

  printf("\n\n\n");
  printf("Best regularization - ALPHA^2: = %.3e\n", a * a);
  printf("Train on full data set...\n\n");
  S = flattenSample(sample, nSeries, &sampleSize, window, -1);
  scaleGrad = -mu / sampleSize;
  scaleReg = -2 * a * a;
  // Gradient descent iterations
  initWeight(model->weight);
  for (n = 0; n < ITER; n++) {
    // if (n % 20 == 0) {
    // printf("Epoch: %d\n", n);

    //}
    setZeroWeight(gradient);
    shuffleSample(S, sampleSize);
    for (i = 0; i < sampleSize; i++) {
      forwardMLP(model, S[i]);
      backPropMLP(model, &S[i][nVariate * window], gradient);
    }
    addScaledWeight(weight, gradient, scaleGrad, scaleReg);
    float risk;
    for (j = 0; j < nSeries; j++) {
      riskJ[j] = meanRisk(model, sample[j], window, nVariate);
    }

    risk = mean(riskJ, nSeries);
    printf("%d %f - \n", n, risk);
  }
  float risk;
  for (j = 0; j < nSeries; j++) {
    riskJ[j] = meanRisk(model, sample[j], window, nVariate);
  }

  risk = mean(riskJ, nSeries);
  printf(" - Training loss: %f - \n", risk);

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
        fscanf(fp, "%f", &(Sample[i]->matrix[r][c]));
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

float floatParse(char *arg) {
  char *end;
  float strParse;

  strParse = strtof(arg, &end);
  errno = 0;

  if (errno != 0 || *end != '\0') {
    perror("Error while converting arg.\n");
    exit(EXIT_FAILURE);
  }
  return strParse;
}

int main(int argc, char *argv[]) {
  int windowSize, nSeries, nVariate;
  int size[LAYERS], nHiddenUnits;
  float mu;
  char *fileName, **names;
  Matrix *sample;
  MLP model;

  if (argc < 5) {
    printf("Provide args: <filename> <window length> <number hidden units> "
           "<learning rate>\n");
    exit(EXIT_FAILURE);
  }

  fileName = argv[1];
  windowSize = intParse(argv[2]);
  nHiddenUnits = intParse(argv[3]);
  mu = floatParse(argv[4]);

  srand(time(NULL));

  sample = readDataFile(fileName, &nSeries, &nVariate, &names);

  size[0] = nVariate * windowSize;
  size[1] = nHiddenUnits;
  size[2] = nVariate;

  model = makeMLP(LAYERS, size);

  trainMLP(model, sample, nSeries, windowSize, ITER, mu);
  // Implement variable learning rate

  float *yhat;
  for (int j = 0; j < sample[0]->rows - windowSize; j++) {
    yhat = forwardMLP(model, sample[0]->matrix[j]);
    for (int i = 0; i < nVariate; i++) {
      printf("yhat: %.5f --- y: %.5f\n", yhat[i],
             sample[0]->matrix[j + windowSize][i]);
    }
    printf("\n");
  }

  printWeight(model->weight);

  for (int i = 0; i < nSeries; i++) {
    freeMatrix(sample[i]);
    free(names[i]);
  }
  free(sample);
  free(names);
  freeMLP(model);

  return 0;
}
