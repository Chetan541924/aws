using System.Collections.Generic;
using System.Text;
using System.Windows;
using System.Windows.Controls;

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace jpmc_genai
{
    public partial class ExecutionLogViewWindow : Window
    {
        private readonly string _executionId;
        private List<ExecutionStepLog> _steps = new();

        public ExecutionLogViewWindow(ExecutionHistoryModel model)
        {
            InitializeComponent();

            // executionId comes from history row
            _executionId = model.ExecutionID;

            Loaded += ExecutionLogViewWindow_Loaded;
        }

        // -------------------------------
        // Load logs from FastAPI
        // -------------------------------
        private async void ExecutionLogViewWindow_Loaded(object sender, RoutedEventArgs e)
        {
            try
            {
                _steps = await FetchExecutionSteps(_executionId);

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
            catch (Exception ex)
            {
                MessageBox.Show(
                    $"Failed to load execution logs.\n\n{ex.Message}",
                    "Execution Logs",
                    MessageBoxButton.OK,
                    MessageBoxImage.Error
                );
            }
        }

        // -------------------------------
        // FastAPI call
        // -------------------------------
        private async Task<List<ExecutionStepLog>> FetchExecutionSteps(string executionId)
        {
            using var client = new HttpClient
            {
                BaseAddress = new Uri("http://127.0.0.1:8002")
            };

            var response = await client.GetAsync($"/executions/{executionId}/steps");
            response.EnsureSuccessStatusCode();

            var json = await response.Content.ReadAsStringAsync();

            return JsonSerializer.Deserialize<List<ExecutionStepLog>>(
                json,
                new JsonSerializerOptions
                {
                    PropertyNameCaseInsensitive = true
                }
            ) ?? new List<ExecutionStepLog>();
        }

        // -------------------------------
        // Row selection → detail pane
        // -------------------------------
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

            sb.AppendLine("Locator Value:");
            sb.AppendLine(
                step.LocatorValue != null
                    ? JsonSerializer.Serialize(step.LocatorValue, new JsonSerializerOptions { WriteIndented = true })
                    : "-"
            );
            sb.AppendLine();

            sb.AppendLine("Input Value:");
            sb.AppendLine(step.InputValue ?? "-");
            sb.AppendLine();

            sb.AppendLine("Frame URL:");
            sb.AppendLine(step.FrameUrl ?? "-");
            sb.AppendLine();

            sb.AppendLine("Executed At:");
            sb.AppendLine(step.ExecutionTs);

            StepDetailsBox.Text = sb.ToString();
        }
    }
}



using System.Collections.Generic;

namespace jpmc_genai
{
    public class ExecutionHistoryModel
    {
        public List<ExecutionStepLog> StepLogs { get; set; }
    }
}


namespace jpmc_genai
{
    public class ExecutionStepLog
    {
        public int StepOrder { get; set; }
        public string GherkinStep { get; set; }
        public string ActionType { get; set; }
        public string Status { get; set; }
        public string Confidence { get; set; }
        public string LocatorStrategy { get; set; }
        public string InputValue { get; set; }
        public string FrameUrl { get; set; }
    }
}


public class ExecutionStepLogModel
{
    public int StepOrder { get; set; }
    public string GherkinStep { get; set; }
    public string ActionType { get; set; }
    public string Status { get; set; }
    public string Confidence { get; set; }

    public string LocatorStrategy { get; set; }
    public object LocatorValue { get; set; }

    public string FrameUrl { get; set; }
    public string InputValue { get; set; }
    public string ExecutionTs { get; set; }
}


