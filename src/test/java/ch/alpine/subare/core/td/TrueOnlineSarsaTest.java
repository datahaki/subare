// code by jph
package ch.alpine.subare.core.td;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.SimpleTestModels;
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
import ch.alpine.subare.util.AssertFail;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;

public class TrueOnlineSarsaTest {
  @Test
  public void testExact() {
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
  public void testLambda() {
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

  @Test
  public void testFailLambda() {
    MonteCarloInterface monteCarloInterface = new GamblerModel(10, RationalScalar.HALF);
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    AssertFail.of(() -> SarsaType.ORIGINAL.trueOnline(SimpleTestModel.INSTANCE, RealScalar.of(2), featureMapper, //
        learningRate, w, new DiscreteStateActionCounter(), null));
  }

  @Test
  public void testFail() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = new GamblerModel(10, RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    AssertFail.of(() -> SarsaType.ORIGINAL.trueOnline(null, RealScalar.of(0.9), featureMapper, //
        learningRate, w, new DiscreteStateActionCounter(), null));
  }
}
