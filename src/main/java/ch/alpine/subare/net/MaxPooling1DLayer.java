// code by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.io.MathematicaFormat;

public class MaxPooling1DLayer implements Layer {
  final int poolSize;
  int[] lastMaxIndices;

  public MaxPooling1DLayer(int poolSize) {
    this.poolSize = poolSize;
  }

  @Override
  public Tensor forward(Tensor input) {
    int outputSize = Math.divideExact(input.length(), poolSize);
    Tensor output = Array.zeros(outputSize);
    lastMaxIndices = new int[outputSize];
    for (int i = 0; i < outputSize; i++) {
      Scalar max = RealScalar.of(-Double.MAX_VALUE);
      int maxIdx = -1;
      for (int j = 0; j < poolSize; j++) {
        int currentIdx = i * poolSize + j;
        if (Scalars.lessThan(max, input.Get(currentIdx))) {
          max = input.Get(currentIdx);
          maxIdx = currentIdx;
        }
      }
      output.set(max, i);
      lastMaxIndices[i] = maxIdx;
    }
    return output;
  }

  @Override
  public Tensor back(Tensor outputGradient) {
    int inputLength = outputGradient.length() * poolSize;
    Tensor inputGradient = Array.zeros(inputLength);
    for (int i = 0; i < outputGradient.length(); i++) {
      // Only the element that was the "max" gets the gradient
      inputGradient.set(outputGradient.Get(i), lastMaxIndices[i]);
    }
    return inputGradient;
  }

  @Override
  public void update() {
    // nothing to do
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("MaxPooling1D", poolSize);
  }
}
