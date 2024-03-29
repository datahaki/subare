// code by jph
package ch.alpine.subare.util;

import java.util.ArrayDeque;
import java.util.Deque;

import ch.alpine.subare.api.DequeDigest;
import ch.alpine.subare.api.StepRecord;

public abstract class DequeDigestAdapter implements DequeDigest {
  @Override // from StepDigest
  public final void digest(StepRecord stepRecord) {
    Deque<StepRecord> deque = new ArrayDeque<>();
    deque.add(stepRecord); // deque holds a single step
    digest(deque);
  }
}
