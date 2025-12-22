When the user checks the checkbox in Account Activity row 3


elif "checks the checkbox" in step_lower or "selects the checkbox" in step_lower:
    action_type = "CHECKBOX"


elif action_type == "CHECKBOX":

    step_lower = step_name.lower()

    # 1. Extract row number
    match = re.search(r"row\s+(\d+)", step_lower)
    if not match:
        raise RuntimeError("CHECKBOX step must specify row number")

    row_index = int(match.group(1)) - 1

    # 2. Resolve frames
    nav_frame, content_frame = resolve_ccs_frames(page)

    # 3. Locate rows (DO NOT CALL locator)
    rows = content_frame.locator(
        "xpath=//div[@id='ACCOUNT_ACTIVITY_Div']//tr[td]"
    )

    row_count = await rows.count()
    if row_index >= row_count:
        raise RuntimeError(f"Row {row_index + 1} out of range. Found {row_count} rows")

    row = rows.nth(row_index)

    # 4. Locate checkbox INSIDE row (NO parentheses)
    checkbox = row.locator(
        "input[type='checkbox']:not([type='hidden'])"
    ).first

    # 5. Interact
    await checkbox.scroll_into_view_if_needed()
    await checkbox.wait_for(state="visible", timeout=10000)

    if not await checkbox.is_checked():
        await checkbox.check(force=True)
        
    logger.info(LogCategory.EXECUTION, f"Rows found: {row_count}")
    logger.info(LogCategory.EXECUTION, f"Target row index: {row_index + 1}")
    
    checkbox_count = await row.locator("input[type='checkbox']").count()
    logger.info(LogCategory.EXECUTION, f"Checkboxes in row: {checkbox_count}")

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Checkbox selected in Account Activity row {row_index + 1}"
    )
