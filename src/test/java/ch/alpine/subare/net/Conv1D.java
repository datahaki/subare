// adapted from gemini
package ch.alpine.subare.net;

import java.util.Random;

class Conv1D {
  double[] weights;
  double bias;
  double[] lastInput;
  double[] weightsGradient;
  double biasGradient;
  int kernelSize;

  public Conv1D(int kernelSize) {
    this.kernelSize = kernelSize;
    this.weights = new double[kernelSize];
    Random rand = new Random();
    for (int i = 0; i < kernelSize; i++) {
      weights[i] = rand.nextGaussian() * 0.1; // Small random weights
    }
    this.bias = 0.0;
  }

  // Forward Pass
  public double[] forward(double[] input) {
    this.lastInput = input;
    int outputSize = input.length - kernelSize + 1;
    double[] output = new double[outputSize];
    for (int i = 0; i < outputSize; i++) {
      double sum = 0;
      for (int j = 0; j < kernelSize; j++) {
        sum += input[i + j] * weights[j];
      }
      output[i] = sum + bias;
    }
    return output;
  }

  // Backward Pass
  public double[] backward(double[] outputGradient, double learningRate) {
    weightsGradient = new double[kernelSize];
    biasGradient = 0;
    double[] inputGradient = new double[lastInput.length];
    for (int i = 0; i < outputGradient.length; i++) {
      double grad = outputGradient[i];
      biasGradient += grad;
      for (int j = 0; j < kernelSize; j++) {
        // Gradient w.r.t weights: input * outputGradient
        weightsGradient[j] += lastInput[i + j] * grad;
        // Gradient w.r.t input: weight * outputGradient
        inputGradient[i + j] += weights[j] * grad;
      }
    }
    updateWeights(learningRate);
    return inputGradient;
  }

  private void updateWeights(double learningRate) {
    for (int i = 0; i < kernelSize; i++) {
      weights[i] -= learningRate * weightsGradient[i];
    }
    bias -= learningRate * biasGradient;
  }
}
