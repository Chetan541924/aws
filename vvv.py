(//div[@id='ACCOUNT_ACTIVITY_Div']//tr[contains(@class,'subPanelContent')])[1]//input[@type='checkbox' and not(@type='hidden')]

elif action_type == "CHECKBOX":

    step_lower = step_name.lower()

    # -------------------------------
    # Extract row number
    # -------------------------------
    match = re.search(r"row\s*(\d+)", step_lower)
    if not match:
        raise RuntimeError(
            "CHECKBOX step must specify row number (e.g. 'row 1')"
        )

    index = int(match.group(1)) - 1  # zero-based index

    # -------------------------------
    # Resolve frames
    # -------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------
    # Locate ALL visible checkboxes
    # -------------------------------
    checkboxes = content_frame.locator(
        "//div[@id='ACCOUNT_ACTIVITY_Div']"
        "//input[@type='checkbox' and not(@type='hidden')]"
    )

    count = await checkboxes.count()

    if count == 0:
        raise RuntimeError("No checkboxes found in Account Activity")

    if index < 0 or index >= count:
        raise RuntimeError(
            f"Checkbox row {index+1} out of range. Total rows: {count}"
        )

    checkbox = checkboxes.nth(index)

    # -------------------------------
    # Scroll + click
    # -------------------------------
    await checkbox.scroll_into_view_if_needed()
    await checkbox.wait_for(state="visible", timeout=15000)

    if not await checkbox.is_checked():
        await checkbox.click(force=True)

    # -------------------------------
    # Post-action delay (visual verify)
    # -------------------------------
    await page.wait_for_timeout(5000)

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Checkbox selected in Account Activity row {index+1}"
    )

    # Pause AFTER click (visual verify)
    # -------------------------------
    await page.wait_for_timeout(5000)

