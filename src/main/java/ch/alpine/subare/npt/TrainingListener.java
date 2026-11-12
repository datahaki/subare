// code by jph
package ch.alpine.subare.npt;

public interface TrainingListener {
  void onProgress(int epoch, double loss);

  void onComplete(TrainingResult result);

  void onError(Throwable error);
}
