// code by jph
package ch.alpine.subare.book.ch04.rental;

import java.awt.Dimension;
import java.util.List;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.util.gfx.StateRaster;
import ch.alpine.subare.util.gfx.StateRasters;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;

/* package */ class CarRentalRaster implements StateRaster {
  private final CarRental carRental;

  public CarRentalRaster(CarRental carRental) {
    this.carRental = carRental;
  }

  @Override
  public DiscreteModel discreteModel() {
    return carRental;
  }

  @Override
  public Dimension dimensionStateRaster() {
    return new Dimension(carRental.maxCars + 1, carRental.maxCars + 1);
  }

  @Override
  public List<Integer> point(Tensor state) {
    return StateRasters.canonicPoint(state);
  }

  @Override
  public Scalar scaleLoss() {
    return RealScalar.ONE;
  }

  @Override
  public Scalar scaleQdelta() {
    return RealScalar.ONE;
  }

  @Override
  public int joinAlongDimension() {
    return 0;
  }

  @Override
  public int magnify() {
    return 4;
  }
}
