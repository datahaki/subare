// code by jph
package ch.alpine.subare.book.ch04.rental;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;
import java.util.Set;

import org.junit.jupiter.api.Test;

import ch.alpine.tensor.RealScalar;
import ch.alpine.tensor.Scalar;
import ch.alpine.tensor.Tensor;
import ch.alpine.tensor.Tensors;
import ch.alpine.tensor.alg.Range;
import ch.alpine.tensor.sca.Chop;

class CarRentalTest {
  @Test
  void testActions() {
    CarRental carRental = new CarRental(20);
    assertEquals(carRental.actions(Tensors.vector(3, 1)), Range.of(-3, 1 + 1));
  }

  @Test
  void testActions2() {
    CarRental carRental = new CarRental(20);
    assertEquals(carRental.actions(Tensors.vector(10, 10)), Range.of(-5, 5 + 1));
  }

  @Test
  void testTransitionProb() {
    CarRental carRental = new CarRental(20);
    Tensor state = Tensors.vector(2, 3);
    Tensor action = RealScalar.of(-1);
    Tensor next = Tensors.vector(3, 2);
    Scalar prob = carRental.transitionProbability(state, action, next);
    Chop._03.requireClose(prob, RealScalar.of(0.01849));
  }

  @Test
  void testTransitionProb1() {
    CarRental carRental = new CarRental(20);
    Tensor state = Tensors.vector(2, 3);
    Tensor action = RealScalar.of(-1);
    Tensor next = Tensors.vector(10, 10);
    Scalar prob = carRental.transitionProbability(state, action, next);
    Chop._09.requireClose(prob, RealScalar.of(2.053630315535358E-7));
  }

  @Test
  void testTransitionsProb() {
    CarRental carRental = new CarRental(10);
    Tensor state = Tensors.vector(3, 2);
    Tensor action = RealScalar.of(0);
    Scalar sum = RealScalar.ZERO;
    for (Tensor next : carRental.transitions(state, action)) {
      // System.out.println("reaching next=" + next);
      Scalar prob = carRental.transitionProbability(state, action, next);
      sum = sum.add(prob);
    }
    // System.out.println("sum=" + sum);
  }

  @Test
  void testExpectedReward() {
    CarRental carRental = new CarRental(10);
    Tensor state = Tensors.vector(3, 2);
    Tensor action = RealScalar.of(0);
    @SuppressWarnings("unused")
    Scalar reward = carRental.expectedReward(state, action);
    // System.out.println("rewardTotal=" + reward);
  }

  @Test
  void testExpectedRewardNext() {
    CarRental carRental = new CarRental(10);
    Tensor state = Tensors.vector(3, 2);
    Tensor action = RealScalar.of(0);
    Tensor next = Tensors.vector(6, 6);
    @SuppressWarnings("unused")
    Scalar reward = carRental.expectedReward(state, action, next);
    // System.out.println("reward=" + reward);
  }

  @Test
  void testMove() {
    CarRental carRental = new CarRental(20);
    Set<Tensor> set = new HashSet<>();
    for (int c = 0; c < 100; ++c) {
      Tensor state = Tensors.vector(10, 10);
      Tensor action = RealScalar.of(2);
      Tensor next = carRental.move(state, action);
      set.add(next);
    }
    assertTrue(30 < set.size());
  }

  @Test
  void testReward() {
    CarRental carRental = new CarRental(20);
    Tensor state = Tensors.vector(10, 10);
    Tensor action = RealScalar.of(2);
    @SuppressWarnings("unused")
    Scalar reward = carRental.reward(state, action, Tensors.vector(12, 8));
    // System.out.println("reward = " + reward);
  }
}
