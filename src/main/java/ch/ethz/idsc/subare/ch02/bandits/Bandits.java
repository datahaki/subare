// code by jph
package ch.ethz.idsc.subare.ch02.bandits;

import java.util.Random;

import ch.ethz.idsc.subare.util.GlobalAssert;
import ch.ethz.idsc.tensor.DoubleScalar;
import ch.ethz.idsc.tensor.RealScalar;
import ch.ethz.idsc.tensor.Scalar;
import ch.ethz.idsc.tensor.Scalars;
import ch.ethz.idsc.tensor.Tensor;
import ch.ethz.idsc.tensor.Tensors;
import ch.ethz.idsc.tensor.alg.Sort;
import ch.ethz.idsc.tensor.red.Mean;
import ch.ethz.idsc.tensor.red.Variance;
import ch.ethz.idsc.tensor.sca.Chop;
import ch.ethz.idsc.tensor.sca.Sqrt;

/** implementation corresponds to Figure 2.1, p. 30 */
class Bandits {
  private static final Random random = new Random();

  // TODO use random variate
  private static Tensor createGaussian(int n) {
    return Tensors.vector(i -> DoubleScalar.of(random.nextGaussian()), n);
  }

  // ---
  private final Tensor prep;
  private Tensor states;

  Bandits(int n) {
    Tensor data = createGaussian(n);
    Scalar mean = (Scalar) Mean.of(data);
    Tensor temp = data.map(x -> x.subtract(mean)).unmodifiable();
    prep = temp.multiply(Sqrt.function.apply((Scalar) Variance.ofVector(temp)).invert());
    GlobalAssert.of(Scalars.isZero((Scalar) Chop.of(Mean.of(prep))));
    GlobalAssert.of( //
        Scalars.isZero((Scalar) Chop.of(Variance.ofVector(prep).subtract(RealScalar.of(1)))));
  }

  Scalar min = RealScalar.ZERO;
  Scalar max = RealScalar.ZERO;

  void pullAll() {
    states = prep.add(createGaussian(prep.length()));
    Tensor sorted = Sort.of(states);
    min = min.add(sorted.Get(0));
    max = max.add(sorted.Get(states.length() - 1));
  }

  Scalar getLever(int k) {
    return states.Get(k);
  }
}
