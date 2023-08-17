package ch.alpine.subare.book.ch04.gambler;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.api.MonteCarloInterface;
import ch.alpine.subare.api.QsaInterface;
import ch.alpine.subare.api.StateActionCounter;
import ch.alpine.subare.api.StepRecord;
import ch.alpine.subare.td.Sarsa;
import ch.alpine.subare.td.SarsaType;
import ch.alpine.subare.util.ConstantLearningRate;
import ch.alpine.subare.util.DefaultLearningRate;
import ch.alpine.subare.util.DiscreteQsa;
import ch.alpine.subare.util.DiscreteStateActionCounter;
import ch.alpine.subare.util.ExactFeatureMapper;
import ch.alpine.subare.util.FeatureMapper;
import ch.alpine.subare.util.FeatureWeight;
import ch.alpine.subare.util.LearningRate;
import ch.alpine.subare.util.PolicyType;
import ch.alpine.subare.util.SimpleTestModel;
import ch.alpine.tensor.RationalScalar;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class GamblerModelTest {
  @Test
  void testActions() {
    GamblerModel gamblerModel = new GamblerModel(100, RealScalar.of(0.4));
    assertEquals(gamblerModel.actions(RealScalar.ZERO), Tensors.vector(0));
    assertEquals(gamblerModel.actions(RealScalar.of(1)), Tensors.vector(1));
    assertEquals(gamblerModel.actions(RealScalar.of(2)), Tensors.vector(1, 2));
    assertEquals(gamblerModel.actions(RealScalar.of(100)), Tensors.vector(0));
  }

  @Test
  void testActions2() {
    assertEquals(new GamblerModel(10, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2, 3));
    assertEquals(new GamblerModel(5, RealScalar.of(0.4)).actions(RealScalar.of(3)), Tensors.vector(1, 2));
  }

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
    Scalar first = learningRate.alpha(new StepRecord(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertEquals(first, RealScalar.ONE);
    sarsa.sac().digest(new StepRecord(state, action, RealScalar.ZERO, state));
    Scalar second = learningRate.alpha(new StepRecord(state, action, RealScalar.ZERO, state), sarsa.sac());
    assertTrue(Scalars.lessThan(second, first));
  }

  @Test
  void testSimple() {
    MonteCarloInterface monteCarloInterface = GamblerModel.createDefault();
    FeatureMapper featureMapper = ExactFeatureMapper.of(monteCarloInterface);
    assertEquals(featureMapper.featureSize(), 2500);
  }
}
