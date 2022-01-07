// code by jph
package ch.alpine.subare.ch05.racetrack;

import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.util.Index;
import ch.alpine.tensor.DoubleScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.alg.Rescale;
import ch.alpine.tensor.img.ColorDataGradients;
import ch.alpine.tensor.img.ImageResize;
import ch.alpine.tensor.img.Raster;
import ch.alpine.tensor.io.ResourceData;

public enum RacetrackHelper {
  ;
  /** @param trackName
   * @param maxSpeed
   * @return
   * @throws Exception if resource associated to trackName does not exist */
  public static Racetrack create(String trackName, int maxSpeed) {
    return new Racetrack(ResourceData.of("/ch05/" + trackName + ".png"), maxSpeed);
  }

  static Tensor render(Racetrack racetrack, DiscreteQsa qsa, Tensor speed, Tensor action) {
    Tensor tensor = racetrack.image().get(Tensor.ALL, Tensor.ALL, 0).map(scalar -> DoubleScalar.INDETERMINATE);
    DiscreteQsa scaled = qsa.create(Rescale.of(qsa.values()).stream());
    for (Tensor state : racetrack.states())
      if (state.length() == 4 && state.extract(2, 4).equals(speed)) {
        Index index = Index.build(racetrack.actions(state));
        if (index.containsKey(action))
          try {
            Scalar sca = scaled.value(state, action);
            int px = Scalars.intValueExact(state.Get(0));
            int py = Scalars.intValueExact(state.Get(1));
            tensor.set(sca, py, px);
          } catch (Exception exception) {
            // ---
          }
      }
    Tensor image = Raster.of(tensor,ColorDataGradients.CLASSIC);
    return ImageResize.nearest(image, 8);
  }
}
