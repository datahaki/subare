// code by jph
package ch.alpine.subare.net;

import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Unprotect;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Flatten;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.sca.Sign;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer implements NetLayer {
  public static LinearLayer of(Distribution d, RandomGenerator randomGenerator, int post, int ante) {
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = RandomVariate.of(d, randomGenerator, post, ante);
    linearLayer.b = Array.zeros(post);
    return linearLayer;
  }

  public static NetLayer xavier(RandomGenerator randomGenerator, int post, int ante) {
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, RealScalar.of(ante).reciprocal());
    LinearLayer linearLayer = new LinearLayer();
    linearLayer.W = RandomVariate.of(distribution, randomGenerator, post, ante);
    linearLayer.b = Array.zeros(post);
    return linearLayer;
  }

  public static NetLayer xavier(int post, int ante) {
    return xavier(ThreadLocalRandom.current(), post, ante);
  }

  public Tensor W;
  public Tensor b;
  // TODO probably obsolete
  public Scalar l2 = RealScalar.ZERO;
  /** input cache */
  Tensor inputCache;
  Tensor outputCache;
  Tensor gW;
  Tensor gb;

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    int ante = Unprotect.dimension1(W);
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, RealScalar.of(ante).reciprocal());
    W = RandomVariate.of(distribution, randomGenerator, Dimensions.of(W));
  }

  @Override
  public Tensor forward(Tensor x) {
    return outputCache = W.dot(inputCache = x).add(b);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    gW = TensorProduct.of(gradOutput, inputCache);
    Sign.requirePositiveOrZero(l2);
    if (Scalars.nonZero(l2))
      gW = gW.subtract(W.multiply(l2));
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
    return y.subtract(outputCache);
  }

  @Override
  public Stream<Scalar> parameters() {
    return Stream.concat(Flatten.scalars(W), Flatten.scalars(b));
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("LinearLayer", W, b, l2);
  }
}
