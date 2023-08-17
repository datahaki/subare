// code by jph
package ch.alpine.subare.core.util.gfx;

import java.awt.Dimension;
import java.awt.Point;
import java.util.List;
import java.util.Objects;
import java.util.function.UnaryOperator;

import ch.alpine.subare.core.api.DiscreteModel;
import ch.alpine.subare.core.api.Policy;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteValueFunctions;
import ch.alpine.subare.core.util.Loss;
import ch.alpine.subare.core.util.Policies;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.sca.Clips;

// TODO SUBARE all non-terminal function should be package visibility
public enum StateActionRasters {
  ;
  /** @param stateActionRaster
   * @param qsa scaled to contain values in the interval [0, 1]
   * @return */
  private static Tensor _render1(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    DiscreteModel discreteModel = stateActionRaster.discreteModel();
    Dimension dimension = stateActionRaster.dimensionStateActionRaster();
    Tensor tensor = Array.of(list -> DoubleScalar.INDETERMINATE, dimension.height, dimension.width);
    for (Tensor state : discreteModel.states())
      for (Tensor action : discreteModel.actions(state)) {
        Point point = stateActionRaster.point(state, action);
        if (Objects.nonNull(point))
          tensor.set(qsa.value(state, action), point.y, point.x);
      }
    // Clip.UNIT.apply(scalar)
    // System.out.println(Pretty.of(tensor));
    // System.exit(0);
    return tensor;
  }

  /** @param stateActionRaster
   * @param qsa scaled to contain values in the interval [0, 1]
   * @return */
  private static Tensor _render(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    return _render(stateActionRaster, qsa, Rescale::of);
  }

  private static Tensor _render(StateActionRaster stateActionRaster, DiscreteQsa qsa, UnaryOperator<Tensor> uo) {
    Tensor tensor = _render1(stateActionRaster, qsa);
    tensor = uo.apply(tensor);
    // System.out.println(Pretty.of(tensor));
    // System.exit(0);
    return tensor.map(ColorDataGradients.CLASSIC);
  }

  private static Tensor _render(StateActionRaster stateActionRaster, Policy policy) {
    return _render(stateActionRaster, Policies.toQsa(stateActionRaster.discreteModel(), policy));
  }

  // ---
  public static Tensor qsa(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    return ImageResize.nearest(_render(stateActionRaster, qsa), stateActionRaster.magnify());
  }

  public static Tensor qsa_rescaled(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    return qsa(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
  }

  public static Tensor qsaPolicy(StateActionRaster stateActionRaster, DiscreteQsa qsa) {
    Tensor image1 = _render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Policy policy = PolicyType.GREEDY.bestEquiprobable(stateActionRaster.discreteModel(), qsa, null);
    Tensor image2 = _render(stateActionRaster, policy);
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 3);
    return ImageResize.nearest( //
        Join.of(dim, image1, Array.zeros(list), image2), stateActionRaster.magnify());
  }

  public static Tensor qsaPolicyRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = _render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    // return ImageResize.nearest(image1, stateActionRaster.magnify());
    // System.out.println(image1.block(Arrays.asList(0, 0), Arrays.asList(2, 2)));
    Policy policy = PolicyType.GREEDY.bestEquiprobable(stateActionRaster.discreteModel(), qsa, null);
    Tensor image2 = _render(stateActionRaster, policy);
    Scalar qdelta = stateActionRaster.scaleQdelta();
    Tensor image3 = _render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, qdelta), tensor -> tensor.map(UnitClip.FUNCTION));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 3);
    return ImageResize.nearest( //
        Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magnify());
  }

  public static Tensor qsaLossRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = _render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    DiscreteQsa loss = Loss.asQsa(stateActionRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().stream() //
        .map(tensor -> tensor.multiply(stateActionRaster.scaleLoss())) //
        .map(tensor -> tensor.map(Clips.unit())));
    Tensor image2 = _render(stateActionRaster, loss);
    Tensor image3 = _render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 1);
    return ImageResize.nearest( //
        Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3), stateActionRaster.magnify());
  }

  // not recommended, use qsaLossRef instead
  static Tensor qsaRef(StateActionRaster stateActionRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = _render(stateActionRaster, DiscreteValueFunctions.rescaled(qsa));
    Scalar qdelta = stateActionRaster.scaleQdelta();
    Tensor image2 = _render(stateActionRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, qdelta));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateActionRaster.joinAlongDimension();
    list.set(dim, 3);
    return ImageResize.nearest( //
        Join.of(dim, image1, Array.zeros(list), image2), stateActionRaster.magnify());
  }
}
