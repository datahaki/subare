// code by jph, fluric
package ch.alpine.subare.util;

import ch.alpine.subare.api.DiscreteModel;
import ch.alpine.subare.api.PolicyExt;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StandardModel;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.VsInterface;

public abstract class PolicyBase implements PolicyExt {
  protected final DiscreteModel discreteModel;
  // ---
  protected StateActionCounter sac;
  protected QsaInterface qsa;

  protected PolicyBase(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
    this.discreteModel = discreteModel;
    this.sac = sac;
    this.qsa = qsa;
  }

  protected PolicyBase(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
    this.discreteModel = standardModel;
    this.sac = sac;
    // might be inefficient or even stale information
    this.qsa = DiscreteUtils.getQsaFromVs(standardModel, vs);
  }

  public final void setQsa(QsaInterface qsa) {
    this.qsa = qsa;
  }

  @Override // from QsaInterfaceSupplier
  public final QsaInterface qsaInterface() {
    return qsa;
  }

  public final void setSac(StateActionCounter sac) {
    this.sac = sac;
  }

  @Override // from StateActionCounterSupplier
  public final StateActionCounter sac() {
    return sac;
  }
}
