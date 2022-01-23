// code by jph
package ch.alpine.subare.ch05.blackjack;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;
import java.util.Objects;

import ch.alpine.subare.core.Policy;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.img.Raster;

/* package */ enum BlackjackHelper {
  ;
  private static final int MAGNIFY = 5;

  // TODO magnify irregular
  public static Tensor render(Blackjack blackjack, Policy policy) {
    BlackjackRaster blackjackRaster = new BlackjackRaster(blackjack);
    Dimension dimension = blackjackRaster.dimensionStateRaster();
    Tensor tensor = Array.of(list -> DoubleScalar.INDETERMINATE, dimension.height, dimension.width);
    for (Tensor state : blackjack.states()) {
      Point point = blackjackRaster.point(state);
      if (Objects.nonNull(point)) {
        Tensor action = RealScalar.ZERO;
        tensor.set(policy.probability(state, action), point.x, point.y);
      }
    }
    return Raster.of(tensor, ColorDataGradients.CLASSIC);
  }

  public static Tensor render(Blackjack blackjack, DiscreteQsa qsa) {
    return StateRasters.vs_rescale(new BlackjackRaster(blackjack), qsa);
  }

  public static Tensor joinAll(Blackjack blackjack, DiscreteQsa qsa) {
    Tensor im1 = render(blackjack, qsa);
    Policy policy = PolicyType.GREEDY.bestEquiprobable(blackjack, qsa, null);
    Tensor im2 = render(blackjack, policy);
    List<Integer> list = Dimensions.of(im1);
    list.set(1, 2);
    return ImageResize.nearest(Join.of(1, im1, Array.zeros(list), im2), MAGNIFY);
  }
}
