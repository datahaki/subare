// code by chatgpt
// adapted by jph
package ch.alpine.subare.net;

import ch.alpine.tensor.Rational;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.ext.Integers;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.d.BernoulliDistribution;
import ch.alpine.tensor.red.Times;
import ch.alpine.tensor.red.Total;

public class DropoutLayer implements Layer {
  public boolean training = true;
  private Scalar maintain = Rational.HALF;
  private Distribution distribution;
  private Tensor mask;

  public DropoutLayer() {
    distribution = BernoulliDistribution.of(maintain);
  }

  @Override
  public Tensor forward(Tensor input) {
    if (!training)
      return input; // no dropout at inference
    int n = input.length();
    Tensor temp = RandomVariate.of(distribution, n);
    int sum = Total.ofVector(temp).number().intValue();
    mask = temp.multiply(Rational.of(n, sum));
    Integers.requireEquals(n, Total.ofVector(mask).number().intValue());
    return Times.of(input, mask);
  }

  @Override
  public Tensor back(Tensor gradOutput) {
    return training //
        ? Times.of(gradOutput, mask)
        : gradOutput;
  }

  @Override
  public void update() {
    // ---
  }

  @Override
  public Tensor error(Tensor y) {
    return null;
  }
}
