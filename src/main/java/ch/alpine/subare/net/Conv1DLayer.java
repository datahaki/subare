// adapted from gemini
// code by jph
package ch.alpine.subare.net;

import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Append;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.fft.ListCorrelate;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.red.Total;

public class Conv1DLayer implements Layer {
  public static Conv1DLayer of(Distribution d, RandomGenerator randomGenerator, int kernelSize) {
    Conv1DLayer conv1dLayer = new Conv1DLayer();
    conv1dLayer.weights = RandomVariate.of(d, randomGenerator, kernelSize);
    return conv1dLayer;
  }

  Tensor weights;
  Scalar b = RealScalar.ZERO;
  /** input cache */
  Tensor inputCache;
  Tensor gW;
  Scalar gb;

  @Override
  public Tensor forward(Tensor x) {
    return ListCorrelate.of(weights, inputCache = x).maps(b::add);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = dW(gradOutput, inputCache);
    gb = Total.ofVector(gradOutput);
    return convsame(gradOutput, weights);
  }

  @Override
  public void update() {
    weights = weights.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }

  @Override
  public Tensor parameters() {
    return Append.of(weights, b);
  }

  public static Tensor dW(Tensor outputGradient, Tensor lastInput) {
    int kernelSize = lastInput.length() - outputGradient.length() + 1;
    Tensor weightsGradient = Array.zeros(kernelSize);
    for (int i = 0; i < outputGradient.length(); i++)
      for (int j = 0; j < kernelSize; j++) {
        // Gradient w.r.t weights: input * outputGradient
        Scalar s = lastInput.Get(i + j).multiply(outputGradient.Get(i));
        weightsGradient.set(s::add, j);
      }
    return weightsGradient;
  }

  public static Tensor convsame(Tensor gradOutput, Tensor weights) {
    int kernelSize = weights.length();
    Tensor inputGradient = Array.zeros(gradOutput.length() + kernelSize - 1);
    for (int i = 0; i < gradOutput.length(); i++)
      for (int j = 0; j < kernelSize; j++) {
        // Gradient w.r.t input: weight * outputGradient
        Scalar s = weights.Get(j).multiply(gradOutput.Get(i));
        inputGradient.set(s::add, i + j);
      }
    return inputGradient;
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("Conv1DLayer", weights, b);
  }
}
