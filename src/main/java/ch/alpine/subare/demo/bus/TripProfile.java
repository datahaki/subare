// code by jph
package ch.alpine.subare.demo.bus;

import ch.alpine.tensor.Scalar;

/* package */ interface TripProfile {
  int length();

  Scalar costPerUnit(int time);

  Scalar unitsDrawn(int time);
}
