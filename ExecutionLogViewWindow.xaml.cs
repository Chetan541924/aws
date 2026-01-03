using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;

namespace jpmc_genai
{
    public partial class ExecutionLogViewWindow : Window
    {
        private readonly List<ExecutionStepLog> _steps;

        public ExecutionLogViewWindow(ExecutionHistoryModel model)
        {
            InitializeComponent();

            _steps = model.StepLogs;

            if (_steps == null || _steps.Count == 0)
            {
                MessageBox.Show(
                    "No execution logs captured for this run.",
                    "Execution Logs",
                    MessageBoxButton.OK,
                    MessageBoxImage.Information
                );
                return;
            }

            LogsGrid.ItemsSource = _steps;
        }

        private void LogsGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            if (LogsGrid.SelectedItem is not ExecutionStepLog step)
                return;

            var sb = new StringBuilder();

            sb.AppendLine("Gherkin Step:");
            sb.AppendLine(step.GherkinStep);
            sb.AppendLine();

            sb.AppendLine("Action:");
            sb.AppendLine(step.ActionType);
            sb.AppendLine();

            sb.AppendLine("Status:");
            sb.AppendLine(step.Status);
            sb.AppendLine();

            sb.AppendLine("Confidence:");
            sb.AppendLine(step.Confidence);
            sb.AppendLine();

            sb.AppendLine("Locator Strategy:");
            sb.AppendLine(step.LocatorStrategy);
            sb.AppendLine();

            sb.AppendLine("Input Value:");
            sb.AppendLine(step.InputValue ?? "-");
            sb.AppendLine();

            sb.AppendLine("Frame URL:");
            sb.AppendLine(step.FrameUrl ?? "-");

            StepDetailsBox.Text = sb.ToString();
        }
    }
}
