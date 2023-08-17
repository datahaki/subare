// code by jph
package ch.alpine.subare.core.td;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
import ch.alpine.subare.core.api.MonteCarloInterface;
import ch.alpine.subare.core.api.QsaInterface;
import ch.alpine.subare.core.api.StateActionCounter;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.DiscreteUtils;
import ch.alpine.subare.core.util.ExactFeatureMapper;
import ch.alpine.subare.core.util.ExploringStarts;
import ch.alpine.subare.core.util.FeatureMapper;
import ch.alpine.subare.core.util.FeatureWeight;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyBase;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;

class TrueOnlineSarsaTest {
  @Test
  void testExact() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
      FeatureWeight w = new FeatureWeight(featureMapper);
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(SimpleTestModel.INSTANCE, RealScalar.ONE, featureMapper, //
          learningRate, w, sac, policy);
      ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
      // DiscreteUtils.print(trueOnlineSarsa.qsa());
      SimpleTestModels._checkExact(trueOnlineSarsa.qsa());
    }
  }

  @Test
  void testLambda() {
    for (SarsaType sarsaType : SarsaType.values()) {
      MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
      FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
      LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
      FeatureWeight w = new FeatureWeight(featureMapper);
      QsaInterface qsa = DiscreteQsa.build(monteCarloInterface);
      StateActionCounter sac = new DiscreteStateActionCounter();
      PolicyBase policy = PolicyType.EGREEDY.bestEquiprobable(monteCarloInterface, qsa, sac);
      TrueOnlineSarsa trueOnlineSarsa = sarsaType.trueOnline(SimpleTestModel.INSTANCE, RealScalar.of(0.9), featureMapper, //
          learningRate, w, sac, policy);
      for (int index = 0; index < 10; ++index) {
        ExploringStarts.batch(monteCarloInterface, policy, trueOnlineSarsa);
      }
      DiscreteUtils.print(trueOnlineSarsa.qsa());
      // TODO SUBARE doesn't work
      // SimpleTestModels._checkClose(qsa);
    }
  }
}
