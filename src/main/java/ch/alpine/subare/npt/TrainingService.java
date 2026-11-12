// code by jph
package ch.alpine.subare.npt;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

import ch.alpine.subare.net.run.NetTrain;

public class TrainingService {
  private final ExecutorService executor = Executors.newSingleThreadExecutor();
  private volatile boolean cancelled;

  public CompletableFuture<TrainingResult> train(NetTrain network, TrainingListener listener, int epochs) {
    cancelled = false;
    return CompletableFuture.supplyAsync(() -> {
      // long start = System.currentTimeMillis();
      // double loss = Double.NaN;
      // int completed = 0;
      // try {
      // for (int epoch = 1; epoch <= epochs; epoch++) {
      // if (cancelled) {
      // break;
      // }
      // // Perform one training epoch
      // // loss = network.trainEpoch();
      // completed = epoch;
      // listener.onProgress(epoch, loss);
      // }
      // } catch (Throwable error) {
      // listener.onError(error);
      // throw new CompletionException(error);
      // }
      return new TrainingResult(0, 0, 0);
    }, executor).whenComplete((result, error) -> {
      if (error == null) {
        // listener.onComplete(result);
      }
    });
  }

  public void cancel() {
    cancelled = true;
  }

  public void shutdown() {
    executor.shutdown();
  }

  static void main() throws InterruptedException, ExecutionException {
    TrainingService trainingService = new TrainingService();
    CompletableFuture<TrainingResult> future = trainingService.train(null, null, 1000);
    TrainingResult trainingResult = future.get();
    IO.println(trainingResult);
    trainingService.executor.close();
  }
}
