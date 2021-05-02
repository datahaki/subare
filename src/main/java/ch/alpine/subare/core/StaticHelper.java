// code by jph
package ch.alpine.subare.core;

import ch.alpine.tensor.red.Total;

enum StaticHelper {
  ;
  public static final DiscountFunction TOTAL = Total::ofVector;
}
