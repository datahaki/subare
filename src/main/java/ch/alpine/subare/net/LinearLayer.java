// code by jph
package ch.alpine.subare.net;

import java.util.List;
import java.util.random.RandomGenerator;
import java.util.stream.Stream;

import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.ConstantArray;
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
  public Tensor W;
  public Tensor b;
  // TODO probably obsolete
  public Scalar l2 = RealScalar.ZERO;
  /** input cache */
  private Tensor inputCache;
  private Tensor outputCache;
  private Tensor gW;
  private Tensor gb;

  public LinearLayer(int d_out, int d_in) {
    W = ConstantArray.of(DoubleScalar.INDETERMINATE, d_out, d_in);
    b = ConstantArray.of(DoubleScalar.INDETERMINATE, d_out);
  }

  @Override
  public void initialize(RandomGenerator randomGenerator) {
    List<Integer> dims = Dimensions.of(W);
    Distribution distribution = NormalDistribution.of(RealScalar.ZERO, Rational.of(2, dims.get(1)));
    W = RandomVariate.of(distribution, randomGenerator, dims);
    b = Array.zeros(dims.get(0));
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
