// code by jph
package ch.alpine.subare.net;

import java.io.Serializable;
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
import ch.alpine.tensor.alg.Transpose;
import ch.alpine.tensor.alg.VectorQ;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.lie.TensorProduct;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Total;
import ch.alpine.tensor.sca.Sign;

/** inspired by
 * <a href="https://reference.wolfram.com/language/ref/LinearLayer.html">LinearLayer</a> */
public class LinearLayer implements NetLayer, Serializable {
  public Tensor W;
  public Tensor b;
  // TODO probably obsolete
  public Scalar l2 = RealScalar.ZERO;
  /** input cache */
  private Tensor inputCache;
  private Tensor outputCache;
  private Tensor gW;
  private Tensor gb;

  public LinearLayer(int d_in, int d_out) {
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
    inputCache = x;
    Tensor result = null;
    if (VectorQ.of(x)) {
      result = W.dot(x).add(b);
    } else {
      result = Tensor.of(x.dot(Transpose.of(W)).stream().map(b::add));
    }
    return outputCache = result;
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    Tensor gradInput = null;
    if (VectorQ.of(gradOutput)) {
      gW = TensorProduct.of(gradOutput, inputCache);
      gb = gradOutput;
      gradInput = gradOutput.dot(W); // vector of length d_in
    } else {
      // IO.println("---");
      { // requires gradOutput to be prescaled with 1/N !!!
        gW = Transpose.of(gradOutput).dot(inputCache);
        gb = Total.of(gradOutput);
      }
      gradInput = gradOutput.dot(W);
    }
    Sign.requirePositiveOrZero(l2);
    if (Scalars.nonZero(l2))
      gW = gW.subtract(W.multiply(l2));
    return gradInput;
  }

  @Override
  public void update() {
    W = W.add(gW);
    b = b.add(gb);
  }

  @Override
  public Tensor error(Tensor y) {
    return y.subtract(outputCache); // vector of length d_out
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
