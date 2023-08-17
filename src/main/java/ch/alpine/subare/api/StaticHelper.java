// code by jph
package ch.alpine.subare.api;

import ch.alpine.tensor.red.Total;

/* package */ enum StaticHelper {
  ;
  public static final DiscountFunction TOTAL = Total::ofVector;
}
