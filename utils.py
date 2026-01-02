CREATE TABLE execution_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    exeid TEXT,
    step_order INTEGER,
    gherkin_step TEXT,
    step_type TEXT,
    action_type TEXT,
    status TEXT,
    error_message TEXT,
    frame_type TEXT,
    frame_url TEXT,
    locator_strategy TEXT,
    locator_value TEXT,
    resolved_element_tag TEXT,
    resolved_element_text TEXT,
    resolved_element_id TEXT,
    resolved_element_name TEXT,
    resolved_element_xpath TEXT,
    input_value TEXT,
    confidence TEXT,
    execution_ts TEXT
);

step_execution_logs = []
step_counter = 0


for step in step_execution_logs:
    await conn.execute(
        """
        INSERT INTO execution_steps (
            exeid, step_order, gherkin_step, step_type, action_type,
            status, error_message,
            frame_type, frame_url,
            locator_strategy, locator_value,
            resolved_element_tag, resolved_element_text,
            resolved_element_id, resolved_element_name,
            resolved_element_xpath,
            input_value, confidence, execution_ts
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            exeid,
            step["step_order"],
            step["gherkin_step"],
            step["step_type"],
            step["action_type"],
            step["status"],
            step["error_message"],
            step["frame_type"],
            step["frame_url"],
            step["locator_strategy"],
            step["locator_value"],
            step["resolved_element_tag"],
            step["resolved_element_text"],
            step["resolved_element_id"],
            step["resolved_element_name"],
            step["resolved_element_xpath"],
            step["input_value"],
            step["confidence"],
            step["execution_ts"]
        )
    )

await conn.commit()
