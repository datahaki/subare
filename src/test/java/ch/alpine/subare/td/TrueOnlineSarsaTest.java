// code by jph
package ch.alpine.subare.td;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import ch.alpine.subare.api.FeatureMapper;
import ch.alpine.subare.api.LearningRate;
import ch.alpine.subare.api.mod.MonteCarloInterface;
import ch.alpine.subare.api.pol.StateActionCounter;
import ch.alpine.subare.api.val.QsaInterface;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.DiscreteUtils;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.ExploringStarts;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.PolicyBase;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.subare.util.SimpleTestModels;
import ch.alpine.tensor.Rational;
import ch.alpine.tensor.RealScalar;

class TrueOnlineSarsaTest {
  @ParameterizedTest
  @EnumSource
  void testExact(SarsaType sarsaType) {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    LearningRate learningRate = ConstantLearningRate.of(Rational.HALF);
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

  @ParameterizedTest
  @EnumSource
  void testLambda(SarsaType sarsaType) {
    MonteCarloInterface monteCarloInterface = SimpleTestModel.INSTANCE;
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    LearningRate learningRate = ConstantLearningRate.of(Rational.HALF);
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
