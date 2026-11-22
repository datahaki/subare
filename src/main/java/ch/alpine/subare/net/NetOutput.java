package ch.alpine.subare.net;

import ch.alpine.tensor.Tensor;

public interface NetOutput {
  /** ErrorThe element-wise or raw difference between a model's
   * prediction and the ground truth target (e.g., error = y - target ).
   * It is vector-valued or tensor-valued per sample.
   * 
   * @param target
   * @return */
  NetLoss error(Tensor target);
}
