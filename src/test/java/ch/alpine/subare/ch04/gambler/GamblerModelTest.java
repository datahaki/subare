package ch.alpine.subare.ch04.gambler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.core.MonteCarloInterface;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.SimpleTestModel;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.subare.core.util.ConstantLearningRate;
import ch.alpine.subare.core.util.DefaultLearningRate;
import ch.alpine.subare.core.util.DiscreteQsa;
import ch.alpine.subare.core.util.DiscreteStateActionCounter;
import ch.alpine.subare.core.util.ExactFeatureMapper;
import ch.alpine.subare.core.util.FeatureMapper;
import ch.alpine.subare.core.util.FeatureWeight;
import ch.alpine.subare.core.util.LearningRate;
import ch.alpine.subare.core.util.PolicyType;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class GamblerModelTest {
  @Test
  void testFailLambda() {
    MonteCarloInterface monteCarloInterface = new GamblerModel(10, RationalScalar.HALF);
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    assertThrows(Exception.class, () -> SarsaType.ORIGINAL.trueOnline(SimpleTestModel.INSTANCE, RealScalar.of(2), featureMapper, //
        learningRate, w, new DiscreteStateActionCounter(), null));
  }

  @Test
  void testFail() {
    LearningRate learningRate = ConstantLearningRate.of(RationalScalar.HALF);
    MonteCarloInterface monteCarloInterface = new GamblerModel(10, RationalScalar.HALF);
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    FeatureWeight w = new FeatureWeight(featureMapper);
    assertThrows(Exception.class, () -> SarsaType.ORIGINAL.trueOnline(null, RealScalar.of(0.9), featureMapper, //
        learningRate, w, new DiscreteStateActionCounter(), null));
  }

  @Test
  void testFirst() {
    LearningRate learningRate = DefaultLearningRate.of(0.9, .51);
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.4));
    QsaInterface qsa = DiscreteQsa.build(gamblerModel);
    StateActionCounter sac = new DiscreteStateActionCounter();
    Sarsa sarsa = SarsaType.ORIGINAL.sarsa(gamblerModel, learningRate, qsa, sac, PolicyType.EGREEDY.bestEquiprobable(gamblerModel, qsa, sac));
    Tensor state = Tensors.vector(1);
    Tensor action = Tensors.vector(0);
    Scalar first = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertEquals(first, RealScalar.ONE);
    sarsa.sac().digest(new StepAdapter(state, action, RealScalar.ZERO, state));
    Scalar second = learningRate.alpha(new StepAdapter(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertTrue(Scalars.lessThan(second, first));
  }

  @Test
  void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }
}
