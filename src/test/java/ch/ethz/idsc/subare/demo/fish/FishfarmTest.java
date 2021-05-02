// code by jph
package ch.ethz.idsc.subare.demo.fish;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;
import junit.framework.TestCase;

public class FishfarmTest extends TestCase {
  public void testActions() {
    Fishfarm fishfarm = new Fishfarm(20, 10);
    Tensor actions = fishfarm.actions(Tensors.vector(2, 6));
    assertEquals(actions, Range.of(0, 6 + 1));
  }

  public void testMove() {
    Fishfarm fishfarm = new Fishfarm(20, 10);
    Tensor next = fishfarm.move(Tensors.vector(2, 10), RealScalar.of(1));
    assertEquals(next, Tensors.vector(3, 9));
  }

  public void testGrowth() {
    int n = 20;
    Fishfarm fishfarm = new Fishfarm(1, n);
    @SuppressWarnings("unused")
    Tensor res = Range.of(0, n + 1).map(fishfarm::growth);
  }
}
