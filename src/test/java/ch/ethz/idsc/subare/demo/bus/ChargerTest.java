// code by jph
package ch.ethz.idsc.subare.demo.bus;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.red.Tally;
import junit.framework.TestCase;

public class ChargerTest extends TestCase {
  public void testSimple() {
    TripProfile tripProfile = new ConstantDrawTrip(16, 2);
    Charger charger = new Charger(tripProfile, 7);
    assertTrue(charger.isTerminal(Tensors.vector(15, 2)));
    Tensor actions = charger.actions(RealScalar.of(0));
    assertEquals(actions.length(), 5);
  }

  public void testDrawn() {
    TripProfile tripProfile = new ConstantDrawTrip(16, 2);
    Charger charger = new Charger(tripProfile, 7);
    final int time = 2;
    Scalar drawn = tripProfile.unitsDrawn(time);
    Tensor res = charger.move(Tensors.vector(2, 3), RealScalar.of(3));
    assertEquals(res.Get(1), RealScalar.of(3 + 3).subtract(drawn));
  }

  public void testCostPerUnit() {
    TripProfile tripProfile = new ConstantDrawTrip(16, 2);
    Tensor costs = Tensors.vector(i -> tripProfile.costPerUnit(i), 10);
    assertEquals(Tally.of(costs).size(), 4);
  }
}
