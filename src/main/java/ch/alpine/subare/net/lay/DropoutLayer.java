// code by chatgpt
// adapted by jph
package ch.alpine.subare.net.lay;

import java.io.Serializable;

import ch.alpine.subare.net.NetLayer;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.red.Total;

// TODO UNFINISHED UNTESTED WORK IN PROGRESS !!!
public class DropoutLayer implements NetLayer, Serializable {
  public boolean training = true;
  private Scalar maintain = Rational.HALF;
  private Distribution distribution;
  private Tensor mask;

  public DropoutLayer() {
    distribution = BernoulliDistribution.of(maintain);
  }

  @Override
  public Tensor forward(Tensor x) {
    if (!training)
      return x; // no dropout at inference
    int n = x.length();
    Tensor temp = RandomVariate.of(distribution, n);
    int sum = Total.ofVector(temp).number().intValue();
    mask = temp.multiply(Rational.of(n, sum));
    Integers.requireEquals(n, Total.ofVector(mask).number().intValue());
    return Times.of(x, mask);
  }

  @Override
  public Tensor forwardBatch(Tensor x) {
    throw new UnsupportedOperationException();
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return training //
        ? Times.of(gradOutput, mask)
        : gradOutput;
  }

  @Override
  public Tensor backBatch(Tensor gradOutput) {
    throw new UnsupportedOperationException();
  }
}
