// code by jph
package ch.alpine.subare.ch04.grid;

import ch.alpine.subare.core.alg.ActionValueIteration;
import ch.alpine.subare.core.alg.ValueIteration;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensors;
import junit.framework.TestCase;

public class GridworldTest extends TestCase {
  public void testVI() {
    Gridworld gridworld = new Gridworld();
    ValueIteration vi = new ValueIteration(gridworld, gridworld);
    vi.untilBelow(RealScalar.of(.0001));
    DiscreteVs vs = vi.vs();
    // vs.print();
    assertEquals(vs.value(Tensors.vector(0, 2)), RealScalar.of(-2));
    assertEquals(vs.value(Tensors.vector(2, 1)), RealScalar.of(-3));
    assertEquals(vs.value(Tensors.vector(2, 3)), RealScalar.of(-1));
    assertEquals(vs.value(Tensors.vector(3, 3)), RealScalar.of(0));
  }

  public void testAVI() {
    Gridworld gridworld = new Gridworld();
    ActionValueIteration avi = ActionValueIteration.of(gridworld);
    avi.untilBelow(RealScalar.of(.0001));
    // ---
    DiscreteQsa qsa = avi.qsa();
    // qsa.print();
    assertEquals(qsa.value(Tensors.vector(0, 1), Tensors.vector(0, 1)), RealScalar.of(-3));
    assertEquals(qsa.value(Tensors.vector(1, 3), Tensors.vector(-1, 0)), RealScalar.of(-4));
    assertEquals(qsa.value(Tensors.vector(3, 3), Tensors.vector(1, 0)), RealScalar.of(0));
  }
}
