When the user checks the checkbox in Account Activity row 3


elif "checks the checkbox" in step_lower or "selects the checkbox" in step_lower:
    action_type = "CHECKBOX"


elif action_type == "CHECKBOX":

    step_lower = step_name.lower()

    match = re.search(r"row\s+(\d+)", step_lower)
    if not match:
        raise RuntimeError("CHECKBOX step must specify row number")

    row_index = int(match.group(1)) - 1

    await page.wait_for_load_state("networkidle")
    nav_frame, content_frame = resolve_ccs_frames(page)

    rows = content_frame.locator(
        "//div[@id='ACCOUNT_ACTIVITY_Div']//tr[td]"
    )

    row_count = await rows.count()
    if row_index >= row_count:
        raise RuntimeError(
            f"Row index {row_index + 1} out of range (found {row_count} rows)"
        )

    row = rows.nth(row_index)

    checkbox = row.locator(
        "input[type='checkbox']:not([type='hidden'])"
    ).first()

    if await checkbox.count() == 0:
        raise RuntimeError(
            f"No checkbox found in Account Activity row {row_index + 1}"
        )

    await checkbox.scroll_into_view_if_needed()
    await checkbox.wait_for(state="visible", timeout=10000)

    if not await checkbox.is_checked():
        await checkbox.check(force=True)

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Checkbox selected in Account Activity row {row_index + 1}"
    )
