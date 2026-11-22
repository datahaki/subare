// code by chatgpt
// adapted by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;
import ch.alpine.tensor.red.Entrywise;

public class DropoutLayer implements NetLayer, Serializable {
  public static NetLayer of(Scalar p) {
    return new DropoutLayer(p);
  }

  public static NetLayer of(Number p) {
    return of(RealScalar.of(p));
  }

  // ---
  private final Scalar p;
  private final Distribution distribution;
  private transient Tensor mask;
  private final Scalar factor;
  public boolean useDropout = true;

  private DropoutLayer(Scalar p) {
    this.p = p;
    Scalar _1_p = RealScalar.ONE.subtract(p);
    distribution = BernoulliDistribution.of(_1_p);
    factor = _1_p.reciprocal();
  }

  @Override
  public Tensor forward(Tensor x) {
    return forwardBatch(x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    if (useDropout) {
      // TODO multiplication is wasteful here!
      mask = RandomVariate.of(distribution, Dimensions.of(X)).multiply(factor);
      return Entrywise.mul().apply(mask, X);
    }
    return X; // no dropout at inference
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return backBatch(gradOutput);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return useDropout //
        ? Entrywise.mul().apply(mask, gradOutput)
        : gradOutput;
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("DropoutLayer", p);
  }
}
