// code by jph
package ch.alpine.subare.core.util;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.subare.core.api.DequeDigest;
import ch.alpine.subare.core.api.StepRecord;

public abstract class DequeDigestAdapter implements DequeDigest {
  @Override // from StepDigest
  public final void digest(StepRecord stepInterface) {
    Deque<StepRecord> deque = new ArrayDeque<>();
    deque.add(stepInterface); // deque holds a single step
    digest(deque);
  }
}
