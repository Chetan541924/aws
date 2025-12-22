elif action_type == "CHECKBOX":

    step_lower = step_name.lower()

    # -------------------------------------------------
    # 1. Extract row number from step
    # -------------------------------------------------
    match = re.search(r"row\s+(\d+)", step_lower)
    if not match:
        raise RuntimeError(
            "CHECKBOX step must specify row number (e.g. 'row 3')"
        )

    # Convert to zero-based index
    row_index = int(match.group(1)) - 1

    # -------------------------------------------------
    # 2. Resolve CCS frames
    # -------------------------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------------------------
    # 3. Locate Account Activity rows
    # -------------------------------------------------
    rows = content_frame.locator(
        "xpath=//div[@id='ACCOUNT_ACTIVITY_Div']//tr[td]"
    )

    row_count = await rows.count()
    if row_count == 0:
        raise RuntimeError("No rows found in Account Activity table")

    if row_index < 0 or row_index >= row_count:
        raise RuntimeError(
            f"Row {row_index + 1} out of range (total rows: {row_count})"
        )

    row = rows.nth(row_index)

    # -------------------------------------------------
    # 4. Locate EXACT checkbox inside that row
    #    (NO parentheses, NO callable locators)
    # -------------------------------------------------
    checkbox = row.locator(
        "input[type='checkbox']:not([type='hidden'])"
    ).first

    # -------------------------------------------------
    # 5. Interact safely
    # -------------------------------------------------
    await checkbox.scroll_into_view_if_needed()
    await checkbox.wait_for(state="visible", timeout=15000)

    if not await checkbox.is_checked():
        await checkbox.check(force=True)

    # -------------------------------------------------
    # 6. Log success
    # -------------------------------------------------
    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Checkbox selected in Account Activity row {row_index + 1}"
    )
