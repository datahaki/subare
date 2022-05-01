// code by jph
package ch.alpine.subare.ch05.infvar;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.alg.ActionValueIteration;
import ch.alpine.subare.core.alg.ValueIteration;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.sca.Abs;

class InfiniteVarianceTest {
  @Test
  public void testActionValueIteration() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    ActionValueIteration avi = ActionValueIteration.of(infiniteVariance);
    avi.untilBelow(RealScalar.of(.00001));
    DiscreteQsa qsa = avi.qsa();
    Scalar diff = qsa.value(InfiniteVariance.BACK, InfiniteVariance.BACK).subtract(RealScalar.ONE);
    assertTrue(Scalars.lessThan(Abs.of(diff), RealScalar.of(.001)));
    assertEquals(qsa.value(InfiniteVariance.BACK, InfiniteVariance.END), RealScalar.ZERO);
    assertEquals(qsa.value(InfiniteVariance.END, InfiniteVariance.END), RealScalar.ZERO);
  }

  @Test
  public void testValueIteration() {
    InfiniteVariance infiniteVariance = new InfiniteVariance();
    ValueIteration vi = new ValueIteration(infiniteVariance);
    vi.untilBelow(RealScalar.of(.00001));
    DiscreteVs vs = vi.vs();
    Scalar diff = vs.value(InfiniteVariance.BACK).subtract(RealScalar.ONE);
    assertTrue(Scalars.lessThan(Abs.of(diff), RealScalar.of(.001)));
    assertEquals(vs.value(InfiniteVariance.END), RealScalar.ZERO);
  }
}
