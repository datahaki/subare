// code by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;
import java.util.random.RandomGenerator;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Throw;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.ext.PackageTestAccess;
import ch.alpine.tensor.io.MathematicaFormat;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.sca.Clips;

/** inverted dropout */
public class DropoutLayer implements NetLayer, Serializable {
  static record DropoutDistribution(double p, Scalar factor) implements Distribution, Serializable {
    @Override
    public Scalar randomVariate(RandomGenerator randomGenerator) {
      return randomGenerator.nextDouble() < p ? factor.zero() : factor;
    }
  }

  public static NetLayer of(Scalar p) {
    return new DropoutLayer(p);
  }

  public static NetLayer of(Number p) {
    return of(RealScalar.of(p));
  }

  @PackageTestAccess
  static Distribution distribution(Scalar p) {
    return new DropoutDistribution(p.number().doubleValue(), RealScalar.ONE.subtract(p).reciprocal());
  }

  // ---
  private final Scalar p;
  private final Distribution distribution;
  private transient Tensor mask;
  /** intended as training will enable dropout */
  private transient boolean enabled = false;

  private DropoutLayer(Scalar p) {
    Clips.unit().requireInside(p);
    Throw.unless(Scalars.lessThan(p, p.one()));
    this.p = p;
    distribution = distribution(p);
  }

  @Override
  public Tensor forward(Tensor x) {
    return forwardBatch(x);
  }

  @Override
  public Tensor forwardBatch(Tensor X) {
    return enabled //
        ? Entrywise.mul().apply(mask = RandomVariate.of(distribution, Dimensions.of(X)), X)
        : X; // no dropout at inference
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return backBatch(gradOutput);
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    return enabled //
        ? Entrywise.mul().apply(mask, gradOutput)
        : gradOutput;
  }

  public void setEnabled(boolean enabled) {
    this.enabled = enabled;
  }

  public boolean isEnabled() {
    return enabled;
  }

  @Override
  public String toString() {
    return MathematicaFormat.concise("DropoutLayer", p, enabled);
  }
}
