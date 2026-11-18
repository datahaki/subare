// code by jph
package ch.alpine.subare.net.reg;

import ch.alpine.subare.net.api.NetRegularization;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;

public enum NetRegularizations {
  ;
  public static NetRegularization create(Scalar l1, Scalar l2) {
    boolean z1 = Scalars.isZero(l1);
    boolean z2 = Scalars.isZero(l2);
    if (!z1 && !z2)
      return new ElasticRegularization(l1, l2);
    return null;
  }
}
