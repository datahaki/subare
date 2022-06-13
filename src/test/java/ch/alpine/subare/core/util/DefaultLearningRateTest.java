// code by jph
package ch.alpine.subare.core.util;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import ch.alpine.subare.ch04.gambler.GamblerModel;
import ch.alpine.subare.core.QsaInterface;
import ch.alpine.subare.core.StateActionCounter;
import ch.alpine.subare.core.adapter.StepAdapter;
import ch.alpine.subare.core.td.Sarsa;
import ch.alpine.subare.core.td.SarsaType;
import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Scalars;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;

class DefaultLearningRateTest {
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
  void testFailFactor() {
    assertThrows(Exception.class, () -> DefaultLearningRate.of(0, 1));
    assertThrows(Exception.class, () -> DefaultLearningRate.of(-1, 1));
  }

  @Test
  void testFailExponent() {
    assertThrows(Exception.class, () -> DefaultLearningRate.of(1, 0.5));
    assertThrows(Exception.class, () -> DefaultLearningRate.of(1, 0.4));
  }
}
