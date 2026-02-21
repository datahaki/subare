// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer implements Layer {
  public static LinearLayer of(Distribution d, RandomGenerator randomGenerator, int post, int ante) {
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = RandomVariate.of(d, randomGenerator, post, ante);
    linearLayer.b = Array.zeros(post);
    return linearLayer;
  }

  public static Layer xavier(RandomGenerator randomGenerator, int post, int ante) {
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, RealScalar.of(ante).reciprocal());
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = RandomVariate.of(distribution, randomGenerator, post, ante);
    linearLayer.b = Array.zeros(post);
    return linearLayer;
  }

  public static Layer xavier(int post, int ante) {
    return xavier(ThreadLocalRandom.current(), post, ante);
  }

  Tensor W;
  Tensor b;
  /** input cache */
  Tensor inputCache;
  Tensor gW;
  Tensor gb;

  @Override
  public Tensor forward(Tensor x) {
    return W.dot(inputCache = x).add(b);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    // IO.println("L L recv " + gradOutput.maps(Round._3));
    gW = TensorProduct.of(gradOutput, inputCache);
    gb = gradOutput;
    return gradOutput.dot(W); // gradInput
  }

  @Override
  public void update() {
    W = W.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    throw new IllegalStateException();
  }

  @Override
  public Tensor parameters() {
    return Flatten.of(W, b);
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b);
  }
}
