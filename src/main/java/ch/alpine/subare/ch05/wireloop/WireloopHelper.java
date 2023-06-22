// code by jph
package ch.alpine.subare.ch05.wireloop;

import java.util.List;

import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.alg.ActionValueIterations;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteVs;
import ch.alpine.subare.core.util.Loss;
import ch.alpine.subare.core.util.gfx.StateRasters;
import ch.alpine.subare.util.RobustArgMax;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.api.TensorScalarFunction;
import ch.alpine.tensor.io.ResourceData;
import ch.alpine.tensor.sca.Chop;
import ch.alpine.tensor.sca.Clips;

public enum WireloopHelper {
  ;
  /** @param trackName
   * @param function
   * @param wireloopReward
   * @return
   * @throws Exception if resource associated to trackName does not exist */
  public static Wireloop create(String trackName, TensorScalarFunction function, WireloopReward wireloopReward) {
    return new Wireloop(ResourceData.of("/ch05/" + trackName + ".png"), function, wireloopReward);
  }

  static Wireloop create(String trackName, TensorScalarFunction function) {
    return create(trackName, function, WireloopReward.freeSteps());
  }

  static DiscreteQsa getOptimalQsa(Wireloop wireloop) {
    return ActionValueIterations.solve(wireloop, RealScalar.of(.0001));
  }

  private static Tensor renderActions(Wireloop wireloop, QsaInterface qsa) {
    WireloopRaster wireloopRaster = new WireloopRaster(wireloop);
    DiscreteVs vs = DiscreteVs.build(wireloop.states());
    RobustArgMax robustArgMax = new RobustArgMax(Chop._06);
    for (Tensor state : wireloop.startStates()) {
      Tensor tensor = Tensor.of(wireloop.actions(state).stream().map(action -> qsa.value(state, action)));
      int index = robustArgMax.of(tensor);
      vs.assign(state, RealScalar.of(index * 0.25 + 0.185));
    }
    return StateRasters.vs(wireloopRaster, vs);
  }

  public static Tensor render(WireloopRaster wireloopRaster, DiscreteQsa ref, DiscreteQsa qsa) {
    Tensor image1 = StateRasters.vs_rescale(wireloopRaster, qsa);
    DiscreteVs loss = Loss.perState(wireloopRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().stream() //
        .map(tensor -> tensor.multiply(wireloopRaster.scaleLoss())) //
        .map(tensor -> tensor.map(Clips.unit())));
    Tensor image2 = StateRasters.vs(wireloopRaster, loss);
    Tensor image3 = renderActions((Wireloop) wireloopRaster.discreteModel(), qsa);
    List<Integer> dimensions = Dimensions.of(image1);
    int dim = wireloopRaster.joinAlongDimension();
    dimensions.set(dim, wireloopRaster.magnify());
    return Join.of(dim, image1, Array.zeros(dimensions), image2, Array.zeros(dimensions), image3);
  }
}
