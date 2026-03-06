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
    conv1dLayer.w = RandomVariate.of(d, randomGenerator, kernelSize);
    return conv1dLayer;
  }

  public Tensor w;
  public Scalar b = RealScalar.ZERO;
  /** input cache */
  Tensor inputCache;
  Tensor outputCache;
  Tensor gW;
  Scalar gb;

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = ListCorrelate.of(w, inputCache = x).maps(b::add);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = dW(gradOutput, inputCache);
    gb = Total.ofVector(gradOutput);
    return convsame(gradOutput, w);
  }

  @Override
  public void update() {
    w = w.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache).multiply(RealScalar.TWO);
  }

  @Override
  public Tensor parameters() {
    return Append.of(w, b);
  }

  private static Tensor dW(Tensor dY, Tensor x) {
    int kernelSize = x.length() - dY.length() + 1;
    Tensor weightsGradient = Array.zeros(kernelSize);
    for (int i = 0; i < dY.length(); i++)
      for (int j = 0; j < kernelSize; j++) {
        // Gradient w.r.t weights: input * outputGradient
        Scalar s = x.Get(i + j).multiply(dY.Get(i));
        weightsGradient.set(s::add, j);
      }
    // IO.println(dY);
    return weightsGradient;
  }

  private static Tensor convsame(Tensor gradOutput, Tensor weights) {
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
    return MathematicaFormat.concise("Conv1DLayer", w, b);
  }
}
