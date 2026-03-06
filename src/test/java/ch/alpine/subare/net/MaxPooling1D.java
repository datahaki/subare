// code by jph
package ch.alpine.subare.net;

public class MaxPooling1D {
  int poolSize;
  int[] lastMaxIndices;

  public MaxPooling1D(int poolSize) {
    this.poolSize = poolSize;
  }

  public double[] forward(double[] input) {
    int outputSize = input.length / poolSize;
    double[] output = new double[outputSize];
    lastMaxIndices = new int[outputSize];
    for (int i = 0; i < outputSize; i++) {
      double max = -Double.MAX_VALUE;
      int maxIdx = -1;
      for (int j = 0; j < poolSize; j++) {
        int currentIdx = i * poolSize + j;
        if (input[currentIdx] > max) {
          max = input[currentIdx];
          maxIdx = currentIdx;
        }
      }
      output[i] = max;
      lastMaxIndices[i] = maxIdx;
    }
    return output;
  }

  public double[] backward(double[] outputGradient, int inputLength) {
    double[] inputGradient = new double[inputLength];
    for (int i = 0; i < outputGradient.length; i++) {
      // Only the element that was the "max" gets the gradient
      inputGradient[lastMaxIndices[i]] = outputGradient[i];
    }
    return inputGradient;
  }
}
