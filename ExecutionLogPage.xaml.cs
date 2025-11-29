using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using jpmc_genai.Services;

namespace jpmc_genai
{
    // =====================================================================
    // MODELS USED IN THIS PAGE
    // =====================================================================


    public class ReusableMethodsResponse
    {
        public string Testcase_Id { get; set; }
        public List<ReusableMethodDto> Results { get; set; }
    }

    public class SimpleScriptResponse
    {
        public string Testcase_Id { get; set; } = "";
        public string Script_Type { get; set; } = "";
        public string Script_Lang { get; set; } = "";
        public string Generated_Script { get; set; } = "";
    }

    // =====================================================================
    // MAIN PAGE
    // =====================================================================
    public partial class ExecutionLogPage : Page
    {
        private readonly ApiClient _apiClient;
        private readonly ObservableCollection<ExecutionLog> _executionLogs;
        private ExecutionLog _selectedExecution;
        private TestPlan _currentTestPlan;

        // REQUIRED FIELD
        private List<ReusableMethodDto> _detectedMethods = new();

        public ExecutionLogPage()
        {
            InitializeComponent();
            _apiClient = new ApiClient();
            _executionLogs = new ObservableCollection<ExecutionLog>();
            ExecutionLogDataGrid.ItemsSource = _executionLogs;
        }

        private void Page_Loaded(object sender, RoutedEventArgs e)
        {
            LoadProjectInfo();
            LoadExecutionLogs();
        }

        private void LoadProjectInfo()
        {
            if (Session.CurrentProject != null)
            {
                ProjectTitleTextBlock.Text = Session.CurrentProject.title;
                ProjectDetailsTextBlock.Text =
                    $"Type: {Session.CurrentProject.projecttype}\nStarted: {Session.CurrentProject.startdate}";
            }
        }

        private async void LoadExecutionLogs()
        {
            try
            {
                StatusTextBlock.Text = "Loading execution history...";
                _apiClient.SetBearer(Session.Token);

                var response = await _apiClient.GetAsync("ExecutionLogs");

                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };

                    var logs = JsonSerializer.Deserialize<List<ExecutionLog>>(content, options);

                    _executionLogs.Clear();
                    foreach (var log in logs.OrderByDescending(l => l.datestamp).ThenByDescending(l => l.exetime))
                        _executionLogs.Add(log);

                    StatusTextBlock.Text = $"Loaded {logs.Count} execution records";
                }
                else
                {
                    StatusTextBlock.Text = $"Error loading logs: {response.StatusCode}";
                }
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"Error: {ex.Message}";
            }
        }

        private void ExecutionLogDataGrid_DoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (ExecutionLogDataGrid.SelectedItem is ExecutionLog log)
            {
                _selectedExecution = log;
                GenerateScriptButton.IsEnabled = true;
                SelectedExecutionInfo.Text =
                    $"Selected: {log.testcaseid} (Execution {log.exeid})";
            }
        }

        // =====================================================================
        //  GENERATE SCRIPT
        // =====================================================================
        private async void GenerateScript_Click(object sender, RoutedEventArgs e)
        {
            if (_selectedExecution == null)
            {
                MessageBox.Show("Please select an execution record first");
                return;
            }

            try
            {
                StatusTextBlock.Text = "Fetching test plan...";
                _apiClient.SetBearer(Session.Token);

                var testPlanResponse = await _apiClient.GetAsync(
                    "TestPlan",
                    new Dictionary<string, string> { { "testCaseId", _selectedExecution.testcaseid } });

                if (!testPlanResponse.IsSuccessStatusCode)
                {
                    StatusTextBlock.Text = "Error fetching test plan";
                    return;
                }

                var testPlanContent = await testPlanResponse.Content.ReadAsStringAsync();
                var options = new JsonSerializerOptions { PropertyNameCaseInsensitive = true };
                _currentTestPlan = JsonSerializer.Deserialize<TestPlan>(testPlanContent, options);

                bool includePrereq = ((ComboBoxItem)PrereqSelector.SelectedItem).Content
                    .ToString().Contains("Include");

                string scriptType =
                    ((ComboBoxItem)ScriptTypeSelector.SelectedItem).Content.ToString().ToLower();

                string scriptLang =
                    ((ComboBoxItem)ScriptLangSelector.SelectedItem).Content.ToString().ToLower();

                StatusTextBlock.Text = "Generating script...";

                var payload = new
                {
                    pretestid_steps = _currentTestPlan?.pretestid_steps ??
                                      new Dictionary<string, Dictionary<string, string>>(),

                    pretestid_scripts = _currentTestPlan?.pretestid_scripts ??
                                        new Dictionary<string, string>(),

                    current_testid = _selectedExecution.testcaseid,

                    current_bdd_steps = _currentTestPlan?.current_bdd_steps ??
                                        new Dictionary<string, string>()
                };

                var jsonContent = new StringContent(
                    JsonSerializer.Serialize(payload),
                    System.Text.Encoding.UTF8,
                    "application/json");

                var scriptResponse = await _apiClient.PostAsync(
                    $"generate-test-script/{_selectedExecution.testcaseid}" +
                    $"?script_type={scriptType}&script_lang={scriptLang}&include_prereq={includePrereq}",
                    jsonContent);

                if (!scriptResponse.IsSuccessStatusCode)
                {
                    MessageBox.Show(await scriptResponse.Content.ReadAsStringAsync());
                    StatusTextBlock.Text = "Script generation failed";
                    return;
                }

                var raw = await scriptResponse.Content.ReadAsStringAsync();

                var result = JsonSerializer.Deserialize<SimpleScriptResponse>(raw, options);

                if (result == null)
                {
                    MessageBox.Show("Failed to parse script:\n" + raw);
                    return;
                }

                ShowSimpleScriptDialog(result.Testcase_Id, result.Generated_Script, _selectedExecution.output);

                StatusTextBlock.Text = "Script generated";
            }
            catch (Exception ex)
            {
                StatusTextBlock.Text = $"Error: {ex.Message}";
                MessageBox.Show(ex.Message);
            }
        }


        // =====================================================================
        // SIMPLE SCRIPT POPUP (WITH NEW BUTTON)
        // =====================================================================
        private void ShowSimpleScriptDialog(string testCaseId, string script, string executionLog)
        {
            var win = new Window
            {
                Title = $"Generated Script - {testCaseId}",
                Width = 900,
                Height = 700,
                Background = Brushes.Black,
                Foreground = Brushes.White,
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                Owner = Window.GetWindow(this)
            };

            var root = new ScrollViewer { VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
            var stack = new StackPanel { Margin = new Thickness(20) };
            root.Content = stack;

            stack.Children.Add(new TextBlock
            {
                Text = "Generated Script",
                FontSize = 16,
                FontWeight = FontWeights.Bold
            });

            stack.Children.Add(new TextBox
            {
                Text = script,
                AcceptsReturn = true,
                IsReadOnly = true,
                Background = Brushes.Black,
                Foreground = Brushes.Cyan,
                FontFamily = new FontFamily("Consolas"),
                Height = 350
            });

            stack.Children.Add(new TextBlock
            {
                Text = "Execution Log",
                FontSize = 16,
                FontWeight = FontWeights.Bold,
                Margin = new Thickness(0, 20, 0, 0)
            });

            stack.Children.Add(new TextBox
            {
                Text = executionLog,
                AcceptsReturn = true,
                IsReadOnly = true,
                Background = Brushes.Black,
                Foreground = Brushes.LightGreen,
                FontFamily = new FontFamily("Consolas"),
                Height = 200
            });

            var btnRow = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 20, 0, 0) };

            var checkBtn = new Button
            {
                Content = "Check Reusable Methods",
                Background = Brushes.Orange,
                Padding = new Thickness(10, 6, 10, 6),
                Margin = new Thickness(0, 0, 10, 0)
            };
            checkBtn.Click += async (s, e) =>
            {
                await CheckReusableMethods(testCaseId, script);
            };

            var downloadBtn = new Button
            {
                Content = "Download Script",
                Background = Brushes.Cyan,
                Padding = new Thickness(10, 6, 10, 6),
                Margin = new Thickness(0, 0, 10, 0)
            };
            downloadBtn.Click += (s, e) => DownloadScript(testCaseId, script);

            var closeBtn = new Button
            {
                Content = "Close",
                Background = Brushes.Gray,
                Padding = new Thickness(10, 6, 10, 6)
            };
            closeBtn.Click += (s, e) => win.Close();

            btnRow.Children.Add(checkBtn);
            btnRow.Children.Add(downloadBtn);
            btnRow.Children.Add(closeBtn);

            stack.Children.Add(btnRow);

            win.Content = root;
            win.ShowDialog();
        }


        // =====================================================================
        // CALL BACKEND TO DETECT REUSABLE METHODS
        // =====================================================================
        private async Task CheckReusableMethods(string testCaseId, string script)
        {
            try
            {
                _apiClient.SetBearer(Session.Token);

                // Extract BDD steps from the loaded TestPlan
                List<string> bddSteps = new();

                if (_currentTestPlan?.current_bdd_steps != null)
                {
                    foreach (var step in _currentTestPlan.current_bdd_steps.Keys)
                        bddSteps.Add(step);
                    // SEND THE STEP TEXT, NOT THE ARGUMENT

                }

                var payload = new
                {
                    testcase_id = testCaseId,
                    generated_script = script,   // IGNORE SCRIPT
                    bdd_steps = bddSteps
                };


                var json = JsonSerializer.Serialize(payload);
                var content = new StringContent(json, System.Text.Encoding.UTF8, "application/json");

                var response = await _apiClient.PostAsync("reusable-methods/check", content);

                if (!response.IsSuccessStatusCode)
                {
                    MessageBox.Show("Error: " + await response.Content.ReadAsStringAsync());
                    return;
                }

                var raw = await response.Content.ReadAsStringAsync();
                MessageBox.Show(raw, "RAW JSON FROM BACKEND");

                var data = JsonSerializer.Deserialize<ReusableMethodsResponse>(
                    raw,
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true }
                );

                if (data?.Results == null || data.Results.Count == 0)
                {
                    MessageBox.Show("No reusable methods detected.");
                    return;
                }

                // 👉 Correct method call:
                var viewWindow = new ReusableMethodViewWindow(data.Results);
                viewWindow.ShowDialog();

            }
            catch (Exception ex)
            {
                MessageBox.Show("Error checking reusable methods:\n" + ex.Message);
            }
        }



        // =====================================================================
        // POPUP WINDOW: DISPLAY REUSABLE METHODS
        // =====================================================================
        private void ShowReusableMethodsWindow(List<ReusableMethodDto> results)
        {
            var win = new Window
            {
                Title = "Reusable Method Suggestions",
                Width = 1100,
                Height = 700,
                Background = Brushes.Black,
                Foreground = Brushes.White,
                WindowStartupLocation = WindowStartupLocation.CenterOwner,
                Owner = Window.GetWindow(this)
            };

            var root = new ScrollViewer();
            var stack = new StackPanel { Margin = new Thickness(20) };
            root.Content = stack;

            stack.Children.Add(new TextBlock
            {
                Text = "Detected Reusable Methods",
                FontSize = 22,
                FontWeight = FontWeights.Bold
            });

            var list = new ListView
            {
                Height = 500,
                Background = Brushes.Black,
                Foreground = Brushes.White,
                BorderBrush = Brushes.Gray,
                BorderThickness = new Thickness(1)
            };

            var gv = new GridView();
            list.View = gv;

            gv.Columns.Add(new GridViewColumn { Header = "Query", Width = 180, DisplayMemberBinding = new System.Windows.Data.Binding("Query") });
            gv.Columns.Add(new GridViewColumn { Header = "Method", Width = 160, DisplayMemberBinding = new System.Windows.Data.Binding("Method_Name") });
            gv.Columns.Add(new GridViewColumn { Header = "Class", Width = 160, DisplayMemberBinding = new System.Windows.Data.Binding("Class_Name") });
            gv.Columns.Add(new GridViewColumn { Header = "Match %", Width = 80, DisplayMemberBinding = new System.Windows.Data.Binding("Match_Percentage") });
            gv.Columns.Add(new GridViewColumn { Header = "Signature", Width = 250, DisplayMemberBinding = new System.Windows.Data.Binding("Full_Signature") });

            list.ItemsSource = results;
            stack.Children.Add(list);

            var codeBox = new TextBox
            {
                AcceptsReturn = true,
                TextWrapping = TextWrapping.NoWrap,
                VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
                HorizontalScrollBarVisibility = ScrollBarVisibility.Auto,
                Background = Brushes.Black,
                Foreground = Brushes.Lime,
                FontFamily = new FontFamily("Consolas"),
                Height = 220,
                Visibility = Visibility.Collapsed
            };

            stack.Children.Add(codeBox);

            list.MouseDoubleClick += (s, e) =>
            {
                if (list.SelectedItem is ReusableMethodDto r)
                {
                    codeBox.Text = r.method_code;
                    codeBox.Visibility = Visibility.Visible;
                }
            };

            var closeBtn = new Button
            {
                Content = "Close",
                Background = Brushes.Gray,
                Foreground = Brushes.White,
                Padding = new Thickness(10, 6, 10, 6),
                Margin = new Thickness(0, 10, 0, 0),
                HorizontalAlignment = HorizontalAlignment.Right
            };
            closeBtn.Click += (s, e) => win.Close();

            stack.Children.Add(closeBtn);

            win.Content = root;
            win.ShowDialog();
        }


        // =====================================================================
        // DOWNLOAD SCRIPT
        // =====================================================================
        private void DownloadScript(string testCaseId, string script)
        {
            try
            {
                var path = Path.Combine(
                    Environment.GetFolderPath(Environment.SpecialFolder.Desktop),
                    $"{testCaseId}_script_{DateTime.Now:yyyyMMdd_HHmmss}.txt");

                File.WriteAllText(path, script);
                MessageBox.Show("Saved:\n" + path);
            }
            catch (Exception ex)
            {
                MessageBox.Show("Error saving file:\n" + ex.Message);
            }
        }


        // =====================================================================
        // NAVIGATION
        // =====================================================================
        private void BackToDashboard_Click(object sender, RoutedEventArgs e)
        {
            if (Session.CurrentProject != null)
                NavigationService.Navigate(new DashboardPage(Session.CurrentProject.projectid));
            else
                NavigationService.Navigate(new ProjectPage());
        }

        private void AITestExecutor_Click(object sender, RoutedEventArgs e)
        {
            if (Session.CurrentProject != null)
                NavigationService.Navigate(new AITestExecutorPage(Session.CurrentProject.projectid));
            else
                MessageBox.Show("Select a project first");
        }

        private void ScriptGenerator_Click(object sender, RoutedEventArgs e)
            => NavigationService.Navigate(new ScriptGeneratorPage());

        private void UploadTestCase_Click(object sender, RoutedEventArgs e)
            => NavigationService.Navigate(new UploadTestCasePage());

        private void ChangeProject_Click(object sender, RoutedEventArgs e)
            => NavigationService.Navigate(new ProjectPage());
    }

    // =====================================================================
    // REUSABLE METHOD DTO — MUST BE OUTSIDE MAIN CLASS
    // =====================================================================
    public class ReusableMethodDto
    {
        public string step_label { get; set; }
        public string step_details { get; set; }
        public string query { get; set; }
        public string method_name { get; set; }
        public string class_name { get; set; }
        public double score { get; set; }
        public string signature { get; set; }
        public string method_code { get; set; }
    }

}
