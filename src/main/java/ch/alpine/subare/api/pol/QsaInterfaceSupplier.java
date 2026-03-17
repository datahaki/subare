// code by jph
package ch.alpine.subare.api.pol;

import ch.alpine.subare.api.val.QsaInterface;

@FunctionalInterface
public interface QsaInterfaceSupplier {
  QsaInterface qsaInterface();
}
