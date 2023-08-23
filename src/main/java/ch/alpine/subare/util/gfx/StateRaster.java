// code by jph
package ch.alpine.subare.util.gfx;

import java.awt.Dimension;
import java.util.List;

import ch.alpine.tensor.Tensor;

public interface StateRaster extends BaseRaster {
  /** @return dimension of raster */
  Dimension dimensionStateRaster();

  /** @param state
   * @return point with x, y as coordinates of state in raster,
   * or null if state does not have a position in the raster */
  List<Integer> point(Tensor state);
}
