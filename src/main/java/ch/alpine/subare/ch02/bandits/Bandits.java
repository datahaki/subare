// code by jph
package ch.alpine.subare.ch02.bandits;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.api.TensorUnaryOperator;
import ch.alpine.tensor.nrm.Normalize;
import ch.alpine.tensor.pdf.Distribution;
import ch.alpine.tensor.pdf.RandomVariate;
import ch.alpine.tensor.pdf.c.NormalDistribution;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.red.ScalarSummaryStatistics;
import ch.alpine.tensor.red.StandardDeviation;
import ch.alpine.tensor.red.Variance;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Clip;
import ch.alpine.tensor.sca.Clips;

/** implementation corresponds to Figure 2.1, p. 30 */
/* package */ class Bandits {
  private static final TensorUnaryOperator NORMALIZE = Normalize.with(StandardDeviation::ofVector);
  private static final Distribution STANDARD = NormalDistribution.standard();
  // ---
  private final Tensor prep;

  public Bandits(int n) {
    Tensor data = RandomVariate.of(STANDARD, n);
    Scalar mean = (Scalar) Mean.of(data);
    prep = NORMALIZE.apply(data.map(x -> x.subtract(mean)));
    Chop._10.requireClose(Mean.of(prep), RealScalar.ZERO);
    Chop._10.requireClose(Variance.ofVector(prep), RealScalar.ONE);
  }

  private Scalar min = RealScalar.ZERO;
  private Scalar max = RealScalar.ZERO;

  Tensor pullAll() {
    Tensor states = prep.add(RandomVariate.of(STANDARD, prep.length()));
    ScalarSummaryStatistics scalarSummaryStatistics = //
        states.stream().map(Scalar.class::cast).collect(ScalarSummaryStatistics.collector());
    min = min.add(scalarSummaryStatistics.getMin());
    max = max.add(scalarSummaryStatistics.getMax());
    return states;
  }

  public Clip clip() {
    return Clips.interval(min, max);
  }
}
