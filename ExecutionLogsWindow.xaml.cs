using System;
using System.Collections.ObjectModel;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;

namespace jpmc_genai
{
    public partial class ExecutionLogsWindow : Window
    {
        private ClientWebSocket? _webSocket;
        private CancellationTokenSource? _cancellationTokenSource;
        private string _testCaseId = "";
        private string _scriptType = "";
        private string _token = "";
        public bool ExecutionCompleted { get; private set; }

        private ObservableCollection<dynamic> _availableMethods = new();
        private bool _awaitingMethodSelection = false;

        public ExecutionLogsWindow()
        {
            InitializeComponent();
            MethodsListBox.ItemsSource = _availableMethods;
        }

        public async Task StartExecution(string testCaseId, string scriptType, string token)
        {
            _testCaseId = testCaseId;
            _scriptType = scriptType;
            _token = token;
            TestCaseInfoTextBlock.Text = $"Test Case: {testCaseId}";
            StatusTextBlock.Text = "Status: Connecting...";
            ExecutionCompleted = false;

            await Task.Delay(100);
            await ConnectAndStreamLogs(testCaseId, scriptType, token);
        }

        private async Task ConnectAndStreamLogs(string testCaseId, string scriptType, string token)
        {
            _webSocket = new ClientWebSocket();
            _cancellationTokenSource = new CancellationTokenSource();

            try
            {
                var wsUri = $"ws://127.0.0.1:8002/testcases/{testCaseId}/execute-with-madl?script_type={scriptType}";

                _webSocket.Options.SetRequestHeader("Authorization", $"Bearer {token}");

                AppendLog($"[INFO] Connecting to: {wsUri}");
                await _webSocket.ConnectAsync(new Uri(wsUri), _cancellationTokenSource.Token);

                StatusTextBlock.Text = "Status: Connected - Execution in progress...";
                AppendLog("[CONNECTED] WebSocket connected successfully");

                await ReceiveLogsAsync();
            }
            catch (WebSocketException ex)
            {
                AppendLog($"[ERROR] WebSocket connection failed: {ex.Message}");
                StatusTextBlock.Text = "Status: Connection failed";
            }
            catch (Exception ex)
            {
                AppendLog($"[ERROR] Failed to connect: {ex.Message}");
                StatusTextBlock.Text = "Status: Connection failed";
            }
        }

        private async Task ReceiveLogsAsync()
        {
            byte[] buffer = new byte[8192];

            try
            {
                while (_webSocket?.State == WebSocketState.Open &&
                       _cancellationTokenSource != null &&
                       !_cancellationTokenSource.Token.IsCancellationRequested)
                {
                    var result = await _webSocket.ReceiveAsync(
                        new ArraySegment<byte>(buffer),
                        _cancellationTokenSource.Token
                    );

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        var json = Encoding.UTF8.GetString(buffer, 0, result.Count);

                        try
                        {
                            using (JsonDocument doc = JsonDocument.Parse(json))
                            {
                                var root = doc.RootElement;

                                if (root.TryGetProperty("status", out var statusProp))
                                {
                                    var status = statusProp.GetString();

                                    // ============================================================
                                    //  ⭐⭐⭐ REQUEST_EDIT HANDLING (FULLY PATCHED)
                                    // ============================================================
                                    if (status == "REQUEST_EDIT")
                                    {
                                        AppendLog("[PLAN] Backend requests testplan editing. Opening editor...");

                                        // FIX #3: Always use GetRawText() for JSON content
                                        string testplanJson = root.GetProperty("testplan").GetRawText();

                                        TestPlanEditorWindow editor = null;
                                        bool editorCrashed = false;

                                        Dispatcher.Invoke(() =>
                                        {
                                            try
                                            {
                                                this.IsEnabled = false;

                                                editor = new TestPlanEditorWindow(testplanJson);
                                                editor.Owner = this;
                                                editor.WindowStartupLocation = WindowStartupLocation.CenterOwner;
                                                editor.ShowDialog();
                                            }
                                            catch (Exception ex)
                                            {
                                                AppendLog("[ERROR] TestPlanEditor crashed: " + ex.Message);
                                                editorCrashed = true;
                                            }
                                            finally
                                            {
                                                this.IsEnabled = true;
                                            }
                                        });

                                        // If loading the editor failed → skip editing 
                                        if (editorCrashed)
                                        {
                                            AppendLog("[PLAN] Editor failed to load. Skipping editing.");

                                            var payloadSkip = new { type = "SKIP_EDIT" };
                                            string skipOut = JsonSerializer.Serialize(payloadSkip);

                                            await _webSocket.SendAsync(
                                                Encoding.UTF8.GetBytes(skipOut),
                                                WebSocketMessageType.Text,
                                                true,
                                                _cancellationTokenSource.Token
                                            );

                                            continue;
                                        }

                                        // User pressed Continue → Modified testplan returned
                                        if (editor != null && !string.IsNullOrEmpty(editor.EditedTestPlanJson))
                                        {
                                            AppendLog("[PLAN] User edited the test plan. Sending edited plan to backend...");

                                            var payload = new
                                            {
                                                type = "EDITED_TESTPLAN",
                                                testplan_json = editor.EditedTestPlanJson
                                            };

                                            string jsonOut = JsonSerializer.Serialize(payload);

                                            await _webSocket.SendAsync(
                                                Encoding.UTF8.GetBytes(jsonOut),
                                                WebSocketMessageType.Text,
                                                true,
                                                _cancellationTokenSource.Token
                                            );
                                        }
                                        else
                                        {
                                            // User pressed Skip Editing
                                            AppendLog("[PLAN] User skipped editing testplan. Informing backend...");

                                            var payload = new { type = "SKIP_EDIT" };
                                            string jsonOut = JsonSerializer.Serialize(payload);

                                            await _webSocket.SendAsync(
                                                Encoding.UTF8.GetBytes(jsonOut),
                                                WebSocketMessageType.Text,
                                                true,
                                                _cancellationTokenSource.Token
                                            );
                                        }

                                        continue;   // Always skip normal log processing for REQUEST_EDIT
                                    }

                                    // ============================================================
                                    // EXISTING STATUS HANDLERS
                                    // ============================================================
                                    if (status == "METHODS_FOUND")
                                    {
                                        if (root.TryGetProperty("methods", out var methodsArray))
                                        {
                                            _availableMethods.Clear();
                                            foreach (var method in methodsArray.EnumerateArray())
                                            {
                                                dynamic methodObj = new System.Dynamic.ExpandoObject();
                                                var dict = (System.Collections.Generic.IDictionary<string, object>)methodObj;

                                                if (method.TryGetProperty("signature", out var sig))
                                                    dict["signature"] = sig.GetString() ?? "";
                                                if (method.TryGetProperty("intent", out var intent))
                                                    dict["intent"] = intent.GetString() ?? "";
                                                if (method.TryGetProperty("match_percentage", out var pct))
                                                    dict["match_display"] = $"Match: {pct.GetDouble():F1}%";

                                                _availableMethods.Add(methodObj);
                                            }

                                            Dispatcher.Invoke(() =>
                                            {
                                                MethodSelectionPanel.Visibility = Visibility.Visible;
                                                MethodsMessageTextBlock.Text = root.TryGetProperty("message", out var msg)
                                                    ? msg.GetString() ?? "Select methods:"
                                                    : "Select methods:";
                                                MethodsListBox.SelectionMode = SelectionMode.Multiple;
                                                _awaitingMethodSelection = true;
                                            });

                                            AppendLog($"[MADL] Found {_availableMethods.Count} reusable methods");
                                        }
                                    }
                                    else if (status == "SEARCHING_MADL")
                                    {
                                        AppendLog("[MADL] Searching for reusable methods...");
                                    }
                                    else if (status == "PLAN_READY")
                                    {
                                        AppendLog("[PLAN] Test plan built successfully");
                                    }
                                    else if (status == "GENERATING")
                                    {
                                        AppendLog("[GENERATION] Generating test script...");
                                    }
                                    else if (status == "EXECUTING")
                                    {
                                        AppendLog("[EXECUTION] Starting script execution...");
                                    }
                                    else if (status == "AUTO_HEALING")
                                    {
                                        AppendLog("[HEALING] Script failed - auto-healing...");
                                    }
                                    else if (status == "COMPLETED")
                                    {
                                        if (root.TryGetProperty("summary", out var summaryProp))
                                        {
                                            AppendLog("\n[SUMMARY]\n" + summaryProp.GetRawText());
                                        }

                                        StatusTextBlock.Text = "Status: Execution completed";
                                        AppendLog("\n[COMPLETED] Execution finished");
                                        ExecutionCompleted = true;
                                        break;
                                    }
                                }

                                // ============================================================
                                //  LOG + ERROR HANDLING
                                // ============================================================
                                if (root.TryGetProperty("log", out var logMessage))
                                    AppendLog(logMessage.GetString());

                                if (root.TryGetProperty("error", out var errorMsg))
                                    AppendLog($"[ERROR] {errorMsg.GetString()}");
                            }
                        }
                        catch (JsonException ex)
                        {
                            AppendLog($"[PARSE ERROR] Failed to parse response: {ex.Message}");
                            AppendLog($"[RAW MESSAGE] {json}");
                        }
                    }

                    if (result.EndOfMessage)
                    {
                        if (_webSocket.State == WebSocketState.CloseReceived)
                        {
                            AppendLog("[INFO] Server closed connection");
                            break;
                        }
                    }
                }
            }
            catch (OperationCanceledException)
            {
                AppendLog("[INFO] Execution cancelled by user");
            }
            catch (Exception ex)
            {
                AppendLog($"[ERROR] Receive error: {ex.Message}");
                StatusTextBlock.Text = "Status: Error receiving logs";
            }
            finally
            {
                await CloseWebSocket();
            }
        }


        private void AppendLog(string message)
        {
            Dispatcher.Invoke(() =>
            {
                LogsTextBox.AppendText(message + Environment.NewLine);
                LogsTextBox.ScrollToEnd();
            });
        }

        private async void ConfirmSelectionButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var selectedMethods = new System.Collections.Generic.List<string>();
                foreach (var item in MethodsListBox.SelectedItems)
                {
                    if (item is System.Dynamic.ExpandoObject eo)
                    {
                        var dict = (System.Collections.Generic.IDictionary<string, object>)eo;
                        if (dict.ContainsKey("signature"))
                            selectedMethods.Add(dict["signature"].ToString());
                    }
                }

                var confirmation = new
                {
                    action = "confirm_selection",
                    selected_methods = selectedMethods
                };

                var json = System.Text.Json.JsonSerializer.Serialize(confirmation);
                await _webSocket.SendAsync(
                    new ArraySegment<byte>(Encoding.UTF8.GetBytes(json)),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );

                AppendLog($"[SELECTION] Confirmed {selectedMethods.Count} methods");
                _awaitingMethodSelection = false;
                Dispatcher.Invoke(() => MethodSelectionPanel.Visibility = Visibility.Hidden);
            }
            catch (Exception ex)
            {
                AppendLog($"[ERROR] Failed to confirm selection: {ex.Message}");
            }
        }

        private async void SkipMethodsButton_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                var skip = new { action = "skip_methods" };
                var json = System.Text.Json.JsonSerializer.Serialize(skip);
                await _webSocket.SendAsync(
                    new ArraySegment<byte>(Encoding.UTF8.GetBytes(json)),
                    WebSocketMessageType.Text,
                    true,
                    CancellationToken.None
                );

                AppendLog("[SELECTION] Skipped method selection");
                _awaitingMethodSelection = false;
                Dispatcher.Invoke(() => MethodSelectionPanel.Visibility = Visibility.Hidden);
            }
            catch (Exception ex)
            {
                AppendLog($"[ERROR] Failed to skip methods: {ex.Message}");
            }
        }

        private async Task CloseWebSocket()
        {
            if (_webSocket != null)
            {
                try
                {
                    if (_webSocket.State == WebSocketState.Open)
                    {
                        await _webSocket.CloseAsync(
                            WebSocketCloseStatus.NormalClosure,
                            "Closing",
                            CancellationToken.None
                        );
                    }
                    _webSocket.Dispose();
                }
                catch { }
            }
        }

        private void CopyLogsButton_Click(object sender, RoutedEventArgs e)
        {
            Clipboard.SetText(LogsTextBox.Text);
            MessageBox.Show("Logs copied to clipboard");
        }

        private void ClearLogsButton_Click(object sender, RoutedEventArgs e)
        {
            LogsTextBox.Clear();
        }

        private void CloseButton_Click(object sender, RoutedEventArgs e)
        {
            Close();
        }

        protected override async void OnClosed(EventArgs e)
        {
            _cancellationTokenSource?.Cancel();
            await CloseWebSocket();
            _cancellationTokenSource?.Dispose();
            base.OnClosed(e);
        }
    }
}
