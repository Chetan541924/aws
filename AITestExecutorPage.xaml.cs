using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Windows;
using System.Windows.Controls;
using jpmc_genai.Services;
using System.Threading.Tasks;

namespace jpmc_genai
{
    public partial class AITestExecutorPage : Page
    {
        private readonly string _projectId;
        private List<TestCase>? _allTestCases;
        private string? _selectedTestCaseId;
        private string? _rawTestPlanJson;
        private TestCase? _editedTestCase;

        private ObservableCollection<TestCaseDisplayModel> _testCasesDisplay;
        private ObservableCollection<ExecutionHistoryModel> _executionHistory;

        public AITestExecutorPage(string projectId)
        {
            InitializeComponent();

            _projectId = projectId;

            _testCasesDisplay = new ObservableCollection<TestCaseDisplayModel>();
            _executionHistory = new ObservableCollection<ExecutionHistoryModel>();

            TestCasesDataGrid.ItemsSource = _testCasesDisplay;
            ExecutionHistoryDataGrid.ItemsSource = _executionHistory;

            LoadProjectDetails();
            LoadTestCases();

            // 🔥 Load actual execution history from API instead of dummy data
            _ = LoadExecutionHistoryFromAPI();
        }

        // ---------------------------------------------------------
        // LOAD TEST PLAN
        // ---------------------------------------------------------
        private async void LoadTestPlanButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedTestCaseId))
            {
                MessageBox.Show("Please select a test case first");
                return;
            }

            try
            {
                using var client = new ApiClient();
                client.SetBearer(Session.Token);

                var response = await client.GetAsync($"testplan/{_selectedTestCaseId}");

                if (!response.IsSuccessStatusCode)
                {
                    MessageBox.Show($"Failed to load test plan: {response.StatusCode}");
                    return;
                }

                var json = await response.Content.ReadAsStringAsync();

                if (string.IsNullOrWhiteSpace(json))
                {
                    MessageBox.Show("Test plan is empty.");
                    return;
                }

                var viewer = new TestPlanViewWindow(json);
                viewer.Owner = Window.GetWindow(this);
                viewer.ShowDialog();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading test plan: {ex.Message}");
            }
        }

        // ---------------------------------------------------------
        // LOAD PROJECT DETAILS
        // ---------------------------------------------------------
        private void LoadProjectDetails()
        {
            var project = Session.CurrentUser?.projects?.FirstOrDefault(p => p.projectid == _projectId);

            ProjectTitleTextBlock.Text = project?.title ?? "Unknown Project";
            ProjectDetailsTextBlock.Text =
                $"ID: {_projectId}\nType: {project?.projecttype ?? "N/A"}\nStarted: {project?.startdate ?? "N/A"}";
        }

        // ---------------------------------------------------------
        // LOAD TEST CASES
        // ---------------------------------------------------------
        private async void LoadTestCases()
        {
            try
            {
                using var client = new ApiClient();
                client.SetBearer(Session.Token);

                var response = await client.GetAsync($"projects/{_projectId}/testcases");

                if (!response.IsSuccessStatusCode)
                    return;

                var jsonContent = await response.Content.ReadAsStringAsync();

                _allTestCases = JsonSerializer.Deserialize<List<TestCase>>(jsonContent,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });

                if (_allTestCases == null)
                    return;

                _testCasesDisplay.Clear();

                foreach (var tc in _allTestCases)
                {
                    _testCasesDisplay.Add(new TestCaseDisplayModel
                    {
                        testcaseid = tc.testcaseid,
                        testdesc = tc.testdesc,
                        Type = "BDD",
                        Status = "Draft",
                        StepCount = 0,
                        Created = DateTime.Now.AddDays(-5).ToString("dd-MMM-yy"),
                        LastUpdated = DateTime.Now.ToString("dd-MMM-yy"),
                        LastExecuted = "-",
                        LastExecutionStatus = "-",
                        HasPrerequisite = !string.IsNullOrWhiteSpace(tc.pretestid),
                        PrerequisiteId = tc.pretestid ?? "-"
                    });
                }

                TestCaseCountTextBlock.Text = $"{_allTestCases.Count} test cases";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Error loading test cases: {ex.Message}");
            }
        }

        // ---------------------------------------------------------
        // SELECT TEST CASE
        // ---------------------------------------------------------
        private void TestCasesDataGrid_MouseDoubleClick(object sender, System.Windows.Input.MouseButtonEventArgs e)
        {
            if (TestCasesDataGrid.SelectedItem is TestCaseDisplayModel row)
            {
                _selectedTestCaseId = row.testcaseid;
                SelectedTestCaseTextBlock.Text = $"Selected: {row.testcaseid} - {row.testdesc}";
            }
        }

        // ---------------------------------------------------------
        // LOAD TEST CASE DETAILS WINDOW
        // ---------------------------------------------------------
        private async void LoadTestDetailsButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedTestCaseId))
            {
                MessageBox.Show("Please select a test case first");
                return;
            }

            var testCase = _allTestCases?.FirstOrDefault(tc => tc.testcaseid == _selectedTestCaseId);

            if (testCase == null)
            {
                MessageBox.Show("Test case not found");
                return;
            }

            var dialog = new TestCaseDetailsDialog(_selectedTestCaseId);
            dialog.Owner = Window.GetWindow(this);
            dialog.ShowDialog();
        }

        // ---------------------------------------------------------
        // EXECUTE TEST CASE
        // ---------------------------------------------------------
        private async void ExecuteButton_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrEmpty(_selectedTestCaseId))
            {
                MessageBox.Show("Please select a test case first");
                return;
            }

            try
            {
                var logsWindow = new ExecutionLogsWindow();
                logsWindow.Owner = Window.GetWindow(this);
                logsWindow.Show();

                await logsWindow.StartExecution(_selectedTestCaseId, "playwright", Session.Token);

                // Refresh execution history after running
                await LoadExecutionHistoryFromAPI();
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Execution error: {ex.Message}");
            }
        }

        // ---------------------------------------------------------
        // LOAD REAL EXECUTION HISTORY FROM API
        // ---------------------------------------------------------
        private async Task LoadExecutionHistoryFromAPI()
        {
            try
            {
                using var client = new ApiClient();
                client.SetBearer(Session.Token);

                var response = await client.GetAsync($"projects/{_projectId}/executions/history?limit=100&offset=0");

                if (!response.IsSuccessStatusCode)
                    return;

                var json = await response.Content.ReadAsStringAsync();

                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                var executionData = JsonSerializer.Deserialize<ExecutionHistoryResponse>(json, options);

                _executionHistory.Clear();

                if (executionData?.executions == null)
                    return;

                foreach (var exec in executionData.executions
                             .OrderByDescending(x => x.datestamp))
                {
                    _executionHistory.Add(new ExecutionHistoryModel
                    {
                        ExecutionID = exec.exeid,
                        ScenarioID = exec.testcaseid,
                        LastExecuted = exec.datestamp,
                        ExecutionStatus = exec.status,
                        ExecutedBy = "System",
                        ExecutedByFull = exec.scripttype,
                        TestStepsCount = 0,
                        TotalTestSteps = 0,
                        TestStepsPassed = 0,
                        Column1 = exec.message,
                        Column2 = exec.output
                    });
                }
            }
            catch (Exception ex)
            {
                Console.WriteLine($"[History Error] {ex.Message}");
            }
        }

        // ---------------------------------------------------------
        // OTHER BUTTON HANDLERS
        // ---------------------------------------------------------
        private void ViewAndExecuteButton_Click(object sender, RoutedEventArgs e)
        {
            MessageBox.Show("View and Execute functionality");
        }

        private void ClearExecutionHistory_Click(object sender, RoutedEventArgs e)
        {
            _executionHistory.Clear();
        }

        private void BackToDashboard_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new DashboardPage(_projectId));
        }

        private void ScriptGenerator_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new ScriptGeneratorPage());
        }

        private void ExecutionLog_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new ExecutionLogPage());
        }

        private void UploadTestCase_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new UploadTestCasePage());
        }

        private void ChangeProject_Click(object sender, RoutedEventArgs e)
        {
            NavigationService.Navigate(new ProjectPage());
        }

        private void ViewLogs_Click(object sender, RoutedEventArgs e)
        {
            if (sender is Button btn && btn.DataContext is ExecutionHistoryModel row)
            {
                var win = new ExecutionLogViewWindow(row);
                win.Owner = Window.GetWindow(this);
                win.ShowDialog();
            }
        }

        private void SelectAllCheckBox_Click(object sender, RoutedEventArgs e)
        {
            bool isChecked = ((CheckBox)sender).IsChecked == true;
            foreach (var item in TestCasesDataGrid.Items)
            {
                if (TestCasesDataGrid.ItemContainerGenerator.ContainerFromItem(item) is DataGridRow row)
                    row.IsSelected = isChecked;
            }
        }

        private void TestCasesDataGrid_SelectionChanged(object sender, SelectionChangedEventArgs e)
        {
            int count = TestCasesDataGrid.SelectedItems.Count;
            SelectedTestCaseTextBlock.Text = count == 0
                ? "No test case selected"
                : $"{count} test case{(count > 1 ? "s" : "")} selected";
        }
    }

    // ---------------------------------------------------------
    // MODELS
    // ---------------------------------------------------------
    public class TestCaseDisplayModel
    {
        public string testcaseid { get; set; }
        public string testdesc { get; set; }
        public string Type { get; set; }
        public string Status { get; set; }
        public int StepCount { get; set; }
        public string Created { get; set; }
        public string LastUpdated { get; set; }
        public string LastExecuted { get; set; }
        public string LastExecutionStatus { get; set; }
        public bool HasPrerequisite { get; set; }
        public string PrerequisiteId { get; set; }
    }

    public class ExecutionHistoryModel
    {
        public string ExecutionID { get; set; }
        public string ScenarioID { get; set; }
        public string LastExecuted { get; set; }
        public string ExecutionStatus { get; set; }
        public string ExecutedBy { get; set; }
        public string ExecutedByFull { get; set; }
        public int TestStepsCount { get; set; }
        public int TotalTestSteps { get; set; }
        public int TestStepsPassed { get; set; }
        public string Column1 { get; set; }
        public string Column2 { get; set; }
    }

    public class ExecutionHistoryResponse
    {
        public int total { get; set; }
        public List<ExecutionData> executions { get; set; }
    }

    public class ExecutionData
    {
        public string exeid { get; set; }
        public string testcaseid { get; set; }
        public string scripttype { get; set; }
        public string datestamp { get; set; }
        public string exetime { get; set; }
        public string status { get; set; }
        public string message { get; set; }
        public string output { get; set; }
    }
}
