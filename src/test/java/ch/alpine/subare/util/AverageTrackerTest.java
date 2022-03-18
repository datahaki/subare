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
import ch.alpine.tensor.red.Mean;
import ch.alpine.tensor.sca.Chop;

public class AverageTrackerTest {
  @Test
  public void testAverage() {
    AverageTracker avg = new AverageTracker();
    avg.track(RealScalar.of(3));
    assertEquals(avg.getScalar(), RealScalar.of(3));
    avg.track(RealScalar.of(1));
    assertEquals(avg.getScalar(), RealScalar.of(2));
    avg.track(RealScalar.of(1));
    Chop._10.requireClose(avg.getScalar(), RealScalar.of(5. / 3));
  }

  @Test
  public void testMean() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.stream().forEach(scalar -> avg.track(scalar));
    assertEquals(avg.getScalar(), Mean.of(vec));
  }

  @Test
  public void testMean2() {
    Tensor vec = Tensors.vector(3, 2, 9, 19, 99, 29, 30);
    AverageTracker avg = new AverageTracker();
    vec.stream().map(Scalar.class::cast).forEach(avg::track);
    assertEquals(avg.getScalar(), Mean.of(vec));
  }

  @Test
  public void testEmpty() {
    AverageTracker avg = new AverageTracker();
    assertTrue(Objects.isNull(avg.get()));
  }
}
