// code by jph
package ch.alpine.subare.util.gfx;

import java.awt.Dimension;
import java.util.List;
import java.util.Objects;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.DiscreteValueFunctions;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.Loss;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.chq.ExactTensorQ;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.Raster;
import ch.alpine.tensor.io.Primitives;
import ch.alpine.tensor.sca.Clips;

public enum StateRasters {
  ;
  public static List<Integer> canonicPoint(Tensor state) {
    ExactTensorQ.require(state);
    return Primitives.toListInteger(state);
  }

  /** @param stateRaster
   * @param vs scaled to contain values in the interval [0, 1]
   * @return */
  private static Tensor _render(StateRaster stateRaster, DiscreteVs vs) {
    DiscreteModel discreteModel = stateRaster.discreteModel();
    Dimension dimension = stateRaster.dimensionStateRaster();
    Tensor tensor = Array.of(_ -> DoubleScalar.INDETERMINATE, dimension.height, dimension.width);
    for (Tensor state : discreteModel.states()) {
      List<Integer> point = stateRaster.point(state);
      if (Objects.nonNull(point))
        tensor.set(vs.value(state), point.get(1), point.get(0));
    }
    return Raster.of(tensor, ColorDataGradients.CLASSIC);
  }

  private static Tensor _vs(StateRaster stateRaster, DiscreteQsa qsa) {
    return _render(stateRaster, DiscreteUtils.createVs(stateRaster.discreteModel(), qsa));
  }

  private static Tensor _vs_rescale(StateRaster stateRaster, DiscreteQsa qsa) {
    DiscreteVs vs = DiscreteUtils.createVs(stateRaster.discreteModel(), qsa);
    return _render(stateRaster, vs.create(Rescale.of(vs.values()).stream()));
  }

  // ---
  public static Tensor vs(StateRaster stateRaster, DiscreteVs vs) {
    return _render(stateRaster, vs);
  }

  public static Tensor vs_rescale(StateRaster stateRaster, DiscreteVs vs) {
    return vs(stateRaster, vs.create(Rescale.of(vs.values()).stream()));
  }

  public static Tensor vs(StateRaster stateRaster, DiscreteQsa qsa) {
    return vs(stateRaster, DiscreteUtils.createVs(stateRaster.discreteModel(), qsa));
  }

  public static Tensor vs_rescale(StateRaster stateRaster, DiscreteQsa qsa) {
    DiscreteVs vs = DiscreteUtils.createVs(stateRaster.discreteModel(), qsa);
    return vs(stateRaster, vs.create(Rescale.of(vs.values()).stream()));
  }

  public static Tensor qsaLossRef(StateRaster stateRaster, DiscreteQsa qsa, DiscreteQsa ref) {
    Tensor image1 = _vs_rescale(stateRaster, DiscreteValueFunctions.rescaled(qsa));
    DiscreteVs loss = Loss.perState(stateRaster.discreteModel(), ref, qsa);
    loss = loss.create(loss.values().stream() //
        .map(tensor -> tensor.multiply(stateRaster.scaleLoss())) //
        .map(tensor -> tensor.maps(Clips.unit())));
    Tensor image2 = _render(stateRaster, loss);
    Tensor image3 = _vs(stateRaster, DiscreteValueFunctions.logisticDifference(qsa, ref, stateRaster.scaleQdelta()));
    List<Integer> list = Dimensions.of(image1);
    int dim = stateRaster.joinAlongDimension();
    list.set(dim, 1);
    return Join.of(dim, image1, Array.zeros(list), image2, Array.zeros(list), image3);
  }
}
