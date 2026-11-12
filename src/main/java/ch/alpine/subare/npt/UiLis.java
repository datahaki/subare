// code by jph
package ch.alpine.subare.npt;

import javax.swing.SwingUtilities;

public class UiLis implements TrainingListener {
  @Override
  public void onProgress(int epoch, double loss) {
    SwingUtilities.invokeLater(() -> {
      // lossLabel.setText(String.format("Epoch %d - Loss: %.6f", epoch, loss));
      // lossChart.addPoint(epoch, loss);
    });
  }

  @Override
  public void onComplete(TrainingResult result) {
    SwingUtilities.invokeLater(() -> {
      // statusLabel.setText("Training complete!");
      // startButton.setEnabled(true);
    });
  }

  @Override
  public void onError(Throwable error) {
    SwingUtilities.invokeLater(() -> {
      // statusLabel.setText("Training failed: " + error.getMessage());
      // startButton.setEnabled(true);
    });
  }
}
