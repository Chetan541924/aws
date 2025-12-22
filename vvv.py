When the user checks the checkbox in Account Activity row 3


elif "checks the checkbox" in step_lower or "selects the checkbox" in step_lower:
    action_type = "CHECKBOX"


elif action_type == "CHECKBOX":

    step_lower = step_name.lower()

    # -------------------------------
    # 1. Extract row number (1-based)
    # -------------------------------
    match = re.search(r"row\s+(\d+)", step_lower)
    if not match:
        raise RuntimeError("CHECKBOX step must specify row number")

    row_index = int(match.group(1)) - 1  # convert to 0-based

    # -------------------------------
    # 2. Resolve frames
    # -------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------
    # 3. Locate Account Activity rows
    # -------------------------------
    rows = content_frame.locator(
        "//div[@id='ACCOUNT_ACTIVITY_Div']//tr[td]"
    )

    row_count = await rows.count()
    if row_index >= row_count:
        raise RuntimeError(
            f"Requested row {row_index + 1}, but only {row_count} rows exist"
        )

    row = rows.nth(row_index)

    # -------------------------------
    # 4. Locate EXACT checkbox in row
    # -------------------------------
    checkbox = row.locator(
        "td input[type='checkbox']:not([type='hidden'])"
    ).first()

    if await checkbox.count() == 0:
        raise RuntimeError("Checkbox not found in selected row")

    # -------------------------------
    # 5. Interact safely
    # -------------------------------
    await checkbox.scroll_into_view_if_needed()
    await checkbox.wait_for(state="visible", timeout=10000)

    if not await checkbox.is_checked():
        await checkbox.check(force=True)

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Checkbox selected in Account Activity row {row_index + 1}"
    )
