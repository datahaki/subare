// code by jph
package ch.alpine.subare.core.adapter;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.subare.core.DequeDigest;
import ch.alpine.subare.core.StepRecord;

public abstract class DequeDigestAdapter implements DequeDigest {
  @Override // from StepDigest
  public final void digest(StepRecord stepInterface) {
    Deque<StepRecord> deque = new ArrayDeque<>();
    deque.add(stepInterface); // deque holds a single step
    digest(deque);
  }
}
