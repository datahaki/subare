// code by jph
package ch.alpine.subare.util.gfx;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.tensor.Scalar;

/* package */ interface BaseRaster {
  /** @return underlying discrete model */
  DiscreteModel discreteModel();

  /** @return loss function scale for visualization */
  Scalar scaleLoss();

  /** @return q function error scale for visualization */
  Scalar scaleQdelta();

  /** @return either 0 or 1 as dimension to join q function, loss, etc. */
  int joinAlongDimension();
}
