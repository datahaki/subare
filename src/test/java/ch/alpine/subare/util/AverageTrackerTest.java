// code by jph
package ch.alpine.subare.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Objects;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.chq.ExactTensorQ;
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Chop;

class AverageTrackerTest {
  @Test
  void testAverage() {
    AverageTracker avg = new AverageTracker();
    avg.track(RealScalar.of(3));
    assertEquals(avg.Get(), RealScalar.of(3));
    avg.track(RealScalar.of(1));
    assertEquals(avg.Get(), RealScalar.of(2));
    avg.track(RealScalar.of(1));
    Chop._10.requireClose(avg.Get(), RealScalar.of(5. / 3));
  }

  @Test
  void testMean() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.forEach(avg::track);
    assertEquals(avg.Get(), Mean.of(vec));
    ExactTensorQ.require(avg.get());
  }

  @Test
  void testMean2() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.stream().map(Scalar.class::cast).forEach(avg::track);
    assertEquals(avg.Get(), Mean.of(vec));
  }

  @Test
  void testEmpty() {
    AverageTracker avg = new AverageTracker();
    assertTrue(Objects.isNull(avg.get()));
  }
}
