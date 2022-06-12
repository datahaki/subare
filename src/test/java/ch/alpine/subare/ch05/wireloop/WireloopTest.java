// code by jph
package ch.alpine.subare.ch05.wireloop;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.awt.Dimension;
import java.util.List;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.red.Entrywise;
import ch.alpine.tensor.red.Max;

class WireloopTest {
  @Test
  public void testSimple() {
    String name = "wirec";
    WireloopReward wireloopReward = WireloopReward.freeSteps();
    wireloopReward = WireloopReward.constantCost();
    Wireloop wireloop = WireloopHelper.create(name, WireloopReward::id_x, wireloopReward);
    assertEquals(Dimensions.of(wireloop.states()), List.of(297, 2));
    assertEquals(wireloop.states().stream().reduce(Entrywise.max()).get(), Tensors.vector(15, 23));
    assertEquals(RealScalar.of(15), //
        wireloop.states().get(Tensor.ALL, 0).stream().reduce(Max::of).get());
    assertEquals(RealScalar.of(23), //
        wireloop.states().get(Tensor.ALL, 1).stream().reduce(Max::of).get());
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    Dimension dimension = wireloopRaster.dimensionStateRaster();
    assertEquals(dimension.width, 16);
    assertEquals(dimension.height, 24);
  }
}
