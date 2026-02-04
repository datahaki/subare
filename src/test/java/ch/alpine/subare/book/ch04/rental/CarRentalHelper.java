// code by jph
package ch.alpine.subare.book.ch04.rental;

import java.util.List;

import ch.alpine.subare.api.Policy;
import ch.alpine.subare.util.DiscreteVs;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.PolicyWrap;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Array;
import ch.alpine.tensor.alg.Dimensions;
import ch.alpine.tensor.alg.Join;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.img.Raster;

/* package */ enum CarRentalHelper {
  ;
  public static Tensor render(CarRental carRental, DiscreteVs vs) {
    // TODO SUBARE use createRaster
    final Tensor tensor = Array.of(_ -> DoubleScalar.INDETERMINATE, 21, 21);
    DiscreteVs scaled = vs.create(Rescale.of(vs.values()).stream());
    for (Tensor state : carRental.states()) {
      Scalar sca = scaled.value(state);
      int x = Scalars.intValueExact(state.Get(0));
      int y = Scalars.intValueExact(state.Get(1));
      tensor.set(sca, x, y);
    }
    Tensor image = Raster.of(tensor, ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 4);
  }

  public static Tensor render(CarRental carRental, Policy policy) {
    final Tensor tensor = Array.of(_ -> DoubleScalar.INDETERMINATE, 21, 21);
    PolicyWrap policyWrap = new PolicyWrap(policy);
    for (Tensor state : carRental.states()) {
      Tensor action = policyWrap.next(state, carRental.actions(state));
      int x = Scalars.intValueExact(state.Get(0));
      int y = Scalars.intValueExact(state.Get(1));
      Scalar sca = RealScalar.of(5).add(action).divide(RealScalar.of(10));
      tensor.set(sca, x, y);
    }
    Tensor image = Raster.of(tensor, ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 4);
  }

  public static Tensor joinAll(CarRental carRental, DiscreteVs vs) {
    Tensor im1 = render(carRental, vs);
    Policy pi = PolicyType.GREEDY.bestEquiprobable(carRental, vs, null);
    Tensor im2 = render(carRental, pi);
    List<Integer> list = Dimensions.of(im1);
    list.set(0, 4 * 2);
    return Join.of(0, im1, Array.zeros(list), im2);
  }
}
