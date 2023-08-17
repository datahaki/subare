// code by jph
package ch.alpine.subare.core.util;

import ch.alpine.subare.core.api.DiscreteModel;
import ch.alpine.subare.core.api.QsaInterface;
import ch.alpine.subare.core.api.StandardModel;
import ch.alpine.subare.core.api.StateActionCounter;
import ch.alpine.subare.core.api.VsInterface;

public enum PolicyType {
  GREEDY {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(discreteModel, qsa, sac);
      eGreedyPolicy.setExplorationRate(ConstantExplorationRate.of(0.0));
      return eGreedyPolicy;
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      EGreedyPolicy eGreedyPolicy = new EGreedyPolicy(standardModel, vs, sac);
      eGreedyPolicy.setExplorationRate(ConstantExplorationRate.of(0.0));
      return eGreedyPolicy;
    }
  },
  EGREEDY {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new EGreedyPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new EGreedyPolicy(standardModel, vs, sac);
    }
  },
  UCB {
    @Override
    public PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac) {
      return new UcbPolicy(discreteModel, qsa, sac);
    }

    @Override
    public PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac) {
      return new UcbPolicy(standardModel, vs, sac);
    }
  };

  public abstract PolicyBase bestEquiprobable(DiscreteModel discreteModel, QsaInterface qsa, StateActionCounter sac);

  public abstract PolicyBase bestEquiprobable(StandardModel standardModel, VsInterface vs, StateActionCounter sac);
}
