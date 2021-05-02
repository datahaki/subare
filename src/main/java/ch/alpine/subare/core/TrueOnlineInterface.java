// code by jph, fluric
package ch.alpine.subare.core;

import ch.alpine.subare.core.td.DoubleTrueOnlineSarsa;
import ch.alpine.subare.core.td.TrueOnlineSarsa;

/** name of interface was chosen because of its use in
 * {@link DoubleTrueOnlineSarsa} and {@link TrueOnlineSarsa}
 * 
 * the main purpose of the interface is to provide two different qsa functions:
 * the discrete qsa and the approximated qsa */
public interface TrueOnlineInterface extends DiscreteQsaSupplier, StepDigest, QsaInterfaceSupplier {
  // ---
}
