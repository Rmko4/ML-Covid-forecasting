// The MIT License (MIT)

// Copyright (c) 2020 Remco Leijenaar

//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in
//  all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.

#include <stdio.h>
#include <stdlib.h>

#include <errno.h>  // errno
#include <float.h>  // FLT_MAX
#include <limits.h> // INT_MAX
#include <math.h>   // sqrtf, exp
#include <time.h>   // time

#define STRLEN 30
#define LAYERS 3
#define MAXA 0.5
#define ITER 5000
#define REGITER 20
#define TESTSIZE 7
#define VALSIZE 14

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
        W->matrix[i][j] = randRange(-0.0001, 0.0001); // Change range
      }
      b[i] = 0;
    }
  }
}

// Expects weights of equal size configuration.
void addScaledWeight(Weight weightTo, Weight weightFrom, float s) {
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
        WTo->matrix[i][j] += s * WFrom->matrix[i][j];
      }
      bTo[i] += s * bFrom[i];
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

float **flattenSample(Matrix *sample, int nSeries, int len, int *size) {
  int rows, n, t, i;
  float **flatSample;

  i = 0;
  flatSample = safeMalloc(len * sizeof(float *));

  for (n = 0; n < nSeries; n++) {
    rows = size[n];
    for (t = 0; t < rows; t++) {
      flatSample[i] = &sample[n]->matrix[t][0];
      i++;
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

void backPropMLP(MLP model, float *y, Weight gradient, float alpha) {
  int layers, rows, columns, i, j, k;
  float *b, **x, **delta;
  Matrix W, G;
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
    W = weight->W[k];
    G = gradient->W[k];
    b = gradient->b[k];
    rows = G->rows;
    columns = G->columns;
    for (i = 0; i < rows; i++) {
      for (j = 0; j < columns; j++) {
        G->matrix[i][j] += delta[k][i] * x[k][j] + 2 * alpha * W->matrix[i][j];
      }
      b[i] += delta[k][i];
    }
  }

  for (k = 0; k < layers - 1; k++) {
    free(delta[k]);
  }
  free(delta);
}

float riskMLP(MLP model, Matrix val, int start, int len, int window,
              int nVariate) {
  int i, end;
  float *yhat, risk;

  risk = 0;
  end = start + len;
  // Calculate validation risk on j.
  for (i = start; i < end; i++) {
    yhat = forwardMLP(model, val->matrix[i]);
    risk += MSE(yhat, val->matrix[i + window], nVariate);
  }
  risk /= len;
  return risk;
}

void trainMLP(MLP model, Matrix *sample, int nSeries, int window, int testSize,
              int valSize, int maxIter, float mu) {
  float upA, maxA, alpha;
  float *b, **x, **delta, **S;
  float *riskRTrain, *riskRVal, *riskJTrain, *riskJVal;
  float scaleGrad, risk;
  int layers, rows, columns;
  int sampleSize, *size;
  int reserve, nVariate, r, i, j, n;
  Weight weight, gradient;

  weight = model->weight;
  layers = weight->layers;
  x = model->x;
  nVariate = weight->size[layers - 1];
  reserve = window + testSize;

  gradient = makeWeight(layers, weight->size);
  size = safeMalloc(nSeries * sizeof(int));

  riskJTrain = safeMalloc(nSeries * sizeof(float));
  riskJVal = safeMalloc(nSeries * sizeof(float));
  riskRTrain = safeMalloc(REGITER * sizeof(float));
  riskRVal = safeMalloc(REGITER * sizeof(float));

  sampleSize = -valSize;
  for (int i = 0; i < nSeries; i++) {
    size[i] = sample[i]->rows - reserve;
    sampleSize += size[i];
  }

  maxA = MAXA;
  upA = maxA / REGITER;
  alpha = 0; // CHANGE BACK LATER TO 0

  // Iterating over degrees of flexibility r
  // r represents linear increases log alpha
  for (r = 0; r < REGITER; r++) {
    printf("\n#####################################\n");
    printf("Regularization - ALPHA^2: = %.3e\n", alpha * alpha);
    printf("#####################################\n");
    // Every iteration leave out part of one series.
    for (j = 0; j < 1; j++) {
      printf("## Fold - K: %d\n", j);
      size[j] -= valSize;
      scaleGrad = -mu / sampleSize;
      S = flattenSample(sample, nSeries, sampleSize, size);

      // Gradient descent iterations
      initWeight(model->weight);
      for (n = 0; n < ITER; n++) {
        // printf("Epoch: %d\n", n);
        setZeroWeight(gradient);
        shuffleSample(S, sampleSize);
        for (i = 0; i < sampleSize; i++) {
          forwardMLP(model, S[i]);
          backPropMLP(model, &S[i][nVariate * window], gradient, alpha);
        }
        addScaledWeight(weight, gradient, scaleGrad);

        /*risk = 0;
        for (i = 0; i < nSeries; i++) {
          risk += riskMLP(model, sample[i], 0, size[i], window, nVariate);
        }
        risk /= nSeries;

        printf(" - Training loss Epoch %d: %f - \n", n, risk);*/
      }
      // Training loss
      riskJTrain[j] = 0;
      for (i = 0; i < nSeries; i++) {
        riskJTrain[j] +=
            riskMLP(model, sample[i], 0, size[i], window, nVariate);
      }
      riskJTrain[j] /= nSeries;

      // Validation loss
      riskJVal[j] =
          riskMLP(model, sample[j], size[j], valSize, window, nVariate);
      size[j] += valSize;
      printf(" - Training loss Fold %d: %f - \n", j, riskJTrain[j]);
      printf(" - Validation loss Fold %d: %f - \n", j, riskJVal[j]);

      free(S);
    }

    // Calculate average risk for regularization r
    riskRTrain[r] = mean(riskJTrain, 1);
    riskRVal[r] = mean(riskJVal, 1); // CHANGE BACK to nSeries

    printf(" - Average Training loss: %f - \n", riskRTrain[r]);
    printf(" - Average Validation loss: %f - \n", riskRVal[r]);
    alpha += upA;
  }

  r = argmin(riskRVal, REGITER); // regularization with minimum average risk
  alpha = r * upA;

  printf("\n\n\n");
  printf("Best regularization - ALPHA^2: = %.3e\n", alpha * alpha);
  printf("Train on full training set...\n\n");

  sampleSize += valSize;
  scaleGrad = -mu / sampleSize;
  S = flattenSample(sample, nSeries, sampleSize, size);

  // Gradient descent iterations
  initWeight(model->weight);
  for (n = 0; n < ITER; n++) {
    setZeroWeight(gradient);
    shuffleSample(S, sampleSize);
    for (i = 0; i < sampleSize; i++) {
      forwardMLP(model, S[i]);
      backPropMLP(model, &S[i][nVariate * window], gradient, alpha);
    }
    addScaledWeight(weight, gradient, scaleGrad);
  }

  risk = 0;
  for (i = 0; i < nSeries; i++) {
    risk += riskMLP(model, sample[i], size[i], testSize, window, nVariate);
  }
  risk /= nSeries;

  printf(" - Final Testing loss: %f - \n", risk);

  freeWeight(gradient);
  free(riskRTrain);
  free(riskJTrain);
  free(riskRVal);
  free(riskJVal);
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

  trainMLP(model, sample, nSeries, windowSize, TESTSIZE, VALSIZE, ITER, mu);

  int i, j, k, start, end;
  float *yhat;

  printf("\nPredictions on testing set:\n");
  for (i = 0; i < nSeries; i++) {
    printf("%s:\n", names[i]);
    start = sample[i]->rows - windowSize - TESTSIZE;
    end = start + TESTSIZE;
    for (j = start; j < end; j++) {
      yhat = forwardMLP(model, sample[i]->matrix[j]);
      for (k = 0; k < nVariate; k++) {
        printf("yhat: %.5f --- y: %.5f\n", yhat[k],
               sample[i]->matrix[j + windowSize][k]);
      }
      printf("\n");
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
